import mysql.connector
import os
import shutil
import datetime
import subprocess
import sys
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
RESTORE_LOG_FILE = os.path.join(LOCAL_RESTORE_DIR, 'restore_selected_log.txt')

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
    
    return sorted(files_with_2025)

def display_files():
    """Display the list of available files with IDs"""
    files = get_files_with_2025()
    
    if not files:
        print("No files found with '2025' in their names.")
        return None
    
    print(f"\nFound {len(files)} files containing '2025' in their names:")
    print("-" * 80)
    
    for idx, filename in enumerate(files, 1):
        file_path = os.path.join(NAS_BACKUP_DIR, filename)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"{idx:3d}. {filename} ({file_size:.2f} MB)")
    
    print("-" * 80)
    return files

def get_user_selection(files):
    """Get user input for which files to restore"""
    if not files:
        return []
    
    print("\nEnter the ID numbers of the schemas you want to restore (comma-separated).")
    print("For example: 1,3,5-7,10")
    print("Or enter 'all' to restore all schemas, or 'q' to quit.")
    
    while True:
        selection = input("\nYour selection: ").strip().lower()
        
        if selection == 'q':
            print("Exiting without restoring any schemas.")
            return []
        
        if selection == 'all':
            return files
            
        try:
            selected_indices = []
            parts = selection.split(',')
            
            for part in parts:
                if '-' in part:
                    # Handle ranges like 5-7
                    start, end = map(int, part.split('-'))
                    if start < 1 or end > len(files):
                        print(f"Invalid range: {part}. Valid range is 1-{len(files)}.")
                        break
                    selected_indices.extend(range(start, end + 1))
                else:
                    # Handle single numbers
                    idx = int(part)
                    if idx < 1 or idx > len(files):
                        print(f"Invalid ID: {idx}. Valid range is 1-{len(files)}.")
                        break
                    selected_indices.append(idx)
            else:
                # This executes if the loop completed without a break
                selected_files = [files[i-1] for i in selected_indices]
                return selected_files
                
            # If we get here, there was an error in the selection
            print("Please try again.")
        
        except ValueError:
            print("Invalid input. Please enter ID numbers separated by commas.")

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
        # Make a connection
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
    """Main function to selectively restore databases with '2025' in their names"""
    try:
        # Initialize log file
        with open(RESTORE_LOG_FILE, 'w') as f:
            f.write(f"Selective restore operation started at {datetime.datetime.now()}\n")
        
        # Display available files and get user selection
        all_files = display_files()
        if not all_files:
            return
            
        selected_files = get_user_selection(all_files)
        if not selected_files:
            return
        
        # Confirm with user
        print(f"\nYou've selected {len(selected_files)} schemas to restore:")
        for idx, filename in enumerate(selected_files, 1):
            print(f"{idx}. {filename}")
        
        confirm = input("\nProceed with restoration? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Restoration cancelled.")
            return
        
        # Process the selected files
        log_message(f"Starting restoration of {len(selected_files)} databases...")
        
        for filename in selected_files:
            process_file(filename)
            
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