import mysql.connector
import os
import shutil
import datetime
import subprocess
import concurrent.futures
import re
from mysql.connector import errorcode

# MySQL Configuration (same as auto_dump_fast.py)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'port': 3306,
}

# NAS Configuration - Source directory for backups
NAS_BACKUP_DIR = "/databk/dumps"

# Local directory for temporary restore files
LOCAL_RESTORE_DIR = '/home/admin2/webapp_2/restore'

# Ensure the local restore directory exists
if not os.path.exists(LOCAL_RESTORE_DIR):
    os.makedirs(LOCAL_RESTORE_DIR)

# Log file for tracking restoration operations
RESTORE_LOG_FILE = os.path.join(LOCAL_RESTORE_DIR, 'restore_log.txt')

def log_message(message):
    """Log a message to both console and log file"""
    print(message)
    with open(RESTORE_LOG_FILE, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

def get_files_with_2025():
    """Get all .sql files containing '2025' in their names from NAS_BACKUP_DIR"""
    if not os.path.exists(NAS_BACKUP_DIR):
        log_message(f"Error: Backup directory {NAS_BACKUP_DIR} does not exist.")
        return []
    
    files_with_2025 = []
    
    for filename in os.listdir(NAS_BACKUP_DIR):
        if '2025' in filename and filename.endswith('.sql'):
            files_with_2025.append(filename)
    
    log_message(f"Found {len(files_with_2025)} files containing '2025' in their names.")
    return files_with_2025

def copy_from_nas(filename):
    """Copy the specified file from NAS to local restore directory"""
    source = os.path.join(NAS_BACKUP_DIR, filename)
    destination = os.path.join(LOCAL_RESTORE_DIR, filename)
    
    log_message(f"Copying {filename} from NAS to local directory...")
    try:
        shutil.copy2(source, destination)
        log_message(f"Successfully copied {filename} to local directory.")
        return destination
    except Exception as e:
        log_message(f"Error copying {filename}: {e}")
        return None

def check_database_exists(cursor, database_name):
    """Check if a database already exists"""
    cursor.execute("SHOW DATABASES")
    existing_databases = [db[0] for db in cursor.fetchall()]
    return database_name in existing_databases

def create_database(cursor, database_name):
    """Create a database if it doesn't exist"""
    try:
        if not check_database_exists(cursor, database_name):
            log_message(f"Creating database {database_name}...")
            cursor.execute(f"CREATE DATABASE `{database_name}`;")
            log_message(f"Database {database_name} created successfully.")
        else:
            log_message(f"Database {database_name} already exists.")
    except Exception as e:
        log_message(f"Error creating database {database_name}: {e}")
        raise

def extract_database_name(filename):
    """Extract the database name from the dump filename"""
    # The filename format appears to be something like: mingyi_Flint_tt03_NA_20MHzDOE_20250113.sql
    # We'll use the whole filename without the .sql extension as the database name
    return os.path.splitext(filename)[0]

def restore_database(local_file_path, database_name):
    """Restore the database from the dump file"""
    command = [
        'mysql',
        f"--host={DB_CONFIG['host']}",
        f"--port={DB_CONFIG['port']}",
        f"--user={DB_CONFIG['user']}",
        f"--password={DB_CONFIG['password']}",
        database_name
    ]
    
    log_message(f"Restoring database {database_name} from {local_file_path}...")
    try:
        process = subprocess.run(
            command,
            stdin=open(local_file_path, 'r'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=600  # 10 minutes timeout
        )
        log_message(f"Successfully restored {database_name}.")
        return True
    except subprocess.TimeoutExpired:
        log_message(f"Restoring {database_name} timed out after 10 minutes.")
        return False
    except subprocess.CalledProcessError as e:
        log_message(f"Error restoring {database_name}: {e.stderr}")
        return False
    except Exception as e:
        log_message(f"Unexpected error restoring {database_name}: {e}")
        return False

def cleanup_local_file(local_file_path):
    """Remove the local copy of the dump file"""
    try:
        os.remove(local_file_path)
        log_message(f"Removed local file {local_file_path}")
    except Exception as e:
        log_message(f"Error removing local file {local_file_path}: {e}")

def process_file(filename):
    """Process a single backup file"""
    try:
        # Make a connection for this thread
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        
        # Copy file from NAS
        local_file_path = copy_from_nas(filename)
        if not local_file_path:
            log_message(f"Skipping {filename} due to copy failure.")
            return
            
        # Extract database name from filename
        database_name = extract_database_name(filename)
        
        # Create the database
        create_database(cursor, database_name)
        
        # Restore the database
        success = restore_database(local_file_path, database_name)
        
        # Clean up
        if success:
            cleanup_local_file(local_file_path)
        
        cursor.close()
        cnx.close()
        
    except Exception as e:
        log_message(f"Error processing file {filename}: {e}")

def main():
    """Main function to restore all databases with '2025' in their names"""
    try:
        # Initialize log file
        with open(RESTORE_LOG_FILE, 'w') as f:
            f.write(f"Restore operation started at {datetime.datetime.now()}\n")
        
        # Get files containing 2025 in their names
        files = get_files_with_2025()
        
        if not files:
            log_message("No files found with '2025' in their names.")
            return
            
        log_message(f"Starting restoration of {len(files)} databases...")
        
        # Use ThreadPoolExecutor to process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(process_file, files)
            
        log_message("Restoration process completed.")
        
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            log_message("Error: Access denied. Check your MySQL username and password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            log_message("Error: Database does not exist.")
        else:
            log_message(f"MySQL Error: {err}")
    except Exception as ex:
        log_message(f"An error occurred: {ex}")

if __name__ == '__main__':
    main() 