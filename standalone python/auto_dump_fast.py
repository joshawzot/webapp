import mysql.connector
import os
import shutil
import datetime
import subprocess  # Add this import to handle external commands
from mysql.connector import errorcode
import stat
import concurrent.futures  # Add this import at the top of your script

# MySQL Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'port': 3306,  # Defaults to 3306 if not specified
}

# NAS Configuration
NAS_BACKUP_DIR = "/mnt/tetramem/dump"  # Example, ensure this is the correct path

# Local Configuration
LOCAL_DUMP_DIR = '/home/admin2/webapp_2/dump'  # Directory for local dumps

# Ensure the local dump directory exists
if not os.path.exists(LOCAL_DUMP_DIR):
    os.makedirs(LOCAL_DUMP_DIR)

# Exceptions - Databases to keep permanently
EXCLUDED_DATABASES = [
    'mysql', 'information_schema', 'performance_schema', 'sys',
    # Add any other databases you want to exclude
    'rwb', 'param'
]

def get_databases(cursor):
    """Retrieve a list of all databases."""
    cursor.execute("SHOW DATABASES;")
    databases = [db[0] for db in cursor.fetchall()]
    print(f"Databases found: {databases}")  # Debugging line
    return databases

def drop_database(cursor, database):
    """Drop the specified database from the MySQL server."""
    print(f"Dropping database {database} from MySQL server...")
    cursor.execute(f"DROP DATABASE `{database}`;")
    print(f"Database {database} has been dropped.")

def dump_database(database):
    """Dump the database to a .sql file in the local dump directory."""
    dump_file = os.path.join(LOCAL_DUMP_DIR, f"{database}.sql")
    command = [
        'mysqldump',
        f"--host={DB_CONFIG['host']}",
        f"--port={DB_CONFIG['port']}",
        f"--user={DB_CONFIG['user']}",
        f"--password={DB_CONFIG['password']}",
        '--single-transaction',
        '--quick',
        '--skip-lock-tables',
        database
    ]
    print(f"Dumping database {database} to {dump_file}...")
    try:
        with open(dump_file, 'w') as outfile:
            subprocess.run(command, stdout=outfile, check=True, timeout=180)
        # Set permissions to allow all users to read/write the dump file
        os.chmod(dump_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        return dump_file
    except subprocess.TimeoutExpired:
        print(f"Dumping database {database} took more than 3 minutes and has been skipped.")
        log_skipped_schema(database)
        return None

def transfer_to_nas(dump_file):
    """Transfer the dump file to the NAS."""
    filename = os.path.basename(dump_file)
    destination = os.path.join(NAS_BACKUP_DIR, filename)
    
    # Ensure the NAS backup directory exists:
    if not os.path.exists(NAS_BACKUP_DIR):
        print(f"Creating directory: {NAS_BACKUP_DIR}")
        try:
            os.makedirs(NAS_BACKUP_DIR, exist_ok=True)
        except PermissionError as e:
            print(f"Permission denied while creating NAS directory {NAS_BACKUP_DIR}: {e}")
            raise

    print(f"Transferring {dump_file} to NAS at {destination}...")
    try:
        # Use subprocess with timeout to copy the file
        subprocess.run(['timeout', '180', 'cp', dump_file, destination], check=True)
        print(f"{dump_file} has been copied to {destination}.")
        os.remove(dump_file)  # Remove the local dump file after successful copy
    except subprocess.TimeoutExpired:
        print(f"Transferring {dump_file} took more than 3 minutes and has been skipped.")
        log_skipped_transfer(dump_file)
    except PermissionError as e:
        print(f"Permission denied while transferring {dump_file} to {destination}: {e}")
        print("Trying with root privileges via 'sudo timeout 180 cp' due to permission issues.")
        try:
            subprocess.run(['sudo', 'timeout', '180', 'cp', dump_file, destination], check=True)
            print("File copied with root privileges.")
            os.remove(dump_file)
        except subprocess.TimeoutExpired:
            print(f"Transferring {dump_file} as root took more than 3 minutes and has been skipped.")
            log_skipped_transfer(dump_file)
        except Exception as e2:
            print(f"Failed to copy as root: {e2}")
            raise
    except Exception as e:
        print(f"Failed to transfer {dump_file} to {destination}: {e}")
        raise

def log_skipped_schema(database):
    """Log the name of the skipped schema to a log file."""
    log_file = os.path.join(LOCAL_DUMP_DIR, 'skipped_schemas.log')
    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now()}: {database}\n")

def log_skipped_transfer(dump_file):
    """Log the name of the dump files that failed to transfer due to timeout."""
    log_file = os.path.join(LOCAL_DUMP_DIR, 'skipped_transfers.log')
    database = os.path.basename(dump_file).replace('.sql', '')
    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now()}: {database}\n")

def process_database(db):
    print(f"\nProcessing database: {db}")
    try:
        # Create a new connection for this thread
        cnx_thread = mysql.connector.connect(**DB_CONFIG)
        cursor_thread = cnx_thread.cursor()

        dump_file = dump_database(db)

        if dump_file:
            transfer_to_nas(dump_file)
            drop_database(cursor_thread, db)
            cnx_thread.commit()
        else:
            print(f"Skipping transfer and drop for database {db} due to dump timeout.")

        cursor_thread.close()
        cnx_thread.close()
    except Exception as e:
        print(f"Error processing database {db}: {e}")

def main():
    try:
        # Get the list of databases
        print("Connecting to MySQL server to get database list...")
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        databases = get_databases(cursor)
        cursor.close()
        cnx.close()

        # Filter databases to process
        databases_to_process = []
        for db in databases:
            if '2025' in db:
                print(f"Skipping database containing '2025': {db}")
                continue

            if db in EXCLUDED_DATABASES:
                print(f"Skipping excluded database: {db}")
                continue

            databases_to_process.append(db)

        # Use ThreadPoolExecutor to process databases in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_database, databases_to_process)

        print("\nAll tasks completed successfully.")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Error: Access denied. Check your MySQL username and password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Error: Database does not exist.")
        else:
            print(err)
    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == '__main__':
    main()