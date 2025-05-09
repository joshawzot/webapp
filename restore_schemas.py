#!/usr/bin/env python3
import mysql.connector
import os
import subprocess
import sys
import argparse
from mysql.connector import errorcode

# MySQL Configuration - same as in auto_dump_fast.py
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'port': 3306,
}

# NAS Configuration - same location as in auto_dump_fast.py
NAS_BACKUP_DIR = "/databk/dumps_2025_3_27"

def get_matching_backup_files(search_pattern):
    """Find all backup files that contain the provided search pattern."""
    try:
        all_backup_files = os.listdir(NAS_BACKUP_DIR)
        matching_files = [f for f in all_backup_files if search_pattern in f and f.endswith('.sql')]
        
        if not matching_files:
            print(f"No backup files found matching pattern '{search_pattern}'")
            return []
            
        print(f"Found {len(matching_files)} backup files matching pattern '{search_pattern}':")
        for f in matching_files:
            print(f"  - {f}")
            
        return matching_files
    except FileNotFoundError:
        print(f"Error: Backup directory {NAS_BACKUP_DIR} not found.")
        return []
    except Exception as e:
        print(f"Error listing backup files: {e}")
        return []

def check_database_exists(cursor, database_name):
    """Check if a database already exists."""
    cursor.execute("SHOW DATABASES")
    databases = [db[0] for db in cursor.fetchall()]
    return database_name in databases

def restore_database(backup_file):
    """Restore a database from a backup file."""
    # Extract database name from backup file name (remove .sql extension)
    database_name = os.path.basename(backup_file).replace('.sql', '')
    backup_path = os.path.join(NAS_BACKUP_DIR, backup_file)
    
    print(f"\nRestoring database '{database_name}' from {backup_path}")
    
    # Check if the database already exists
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        
        if check_database_exists(cursor, database_name):
            choice = input(f"Database '{database_name}' already exists. Override? (y/n): ").lower()
            if choice == 'y':
                print(f"Dropping existing database '{database_name}'...")
                cursor.execute(f"DROP DATABASE `{database_name}`")
                cnx.commit()
            else:
                print(f"Skipping restoration of database '{database_name}'")
                cursor.close()
                cnx.close()
                return False
        
        # Create database
        print(f"Creating database '{database_name}'...")
        cursor.execute(f"CREATE DATABASE `{database_name}`")
        cnx.commit()
        cursor.close()
        cnx.close()
        
        # Restore from backup file
        command = [
            'mysql',
            f"--host={DB_CONFIG['host']}",
            f"--port={DB_CONFIG['port']}",
            f"--user={DB_CONFIG['user']}",
            f"--password={DB_CONFIG['password']}",
            database_name
        ]
        
        print(f"Importing data into '{database_name}'...")
        try:
            with open(backup_path, 'r') as infile:
                process = subprocess.run(
                    command,
                    stdin=infile,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
            print(f"Database '{database_name}' has been successfully restored.")
            return True
        except subprocess.TimeoutExpired:
            print(f"Error: Restoration of '{database_name}' timed out after 5 minutes.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Error restoring database '{database_name}': {e}")
            print(f"Error details: {e.stderr}")
            return False
            
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Error: Access denied. Check your MySQL username and password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Error: Database does not exist.")
        else:
            print(f"MySQL Error: {err}")
        return False
    except Exception as e:
        print(f"Error creating database '{database_name}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Restore MySQL databases based on a search pattern')
    parser.add_argument('pattern', help='Search pattern (case sensitive) to match database names')
    args = parser.parse_args()
    
    search_pattern = args.pattern
    
    print(f"Searching for backups containing pattern: '{search_pattern}'")
    matching_files = get_matching_backup_files(search_pattern)
    
    if not matching_files:
        sys.exit(1)
    
    total = len(matching_files)
    restore_count = 0
    
    for i, backup_file in enumerate(matching_files, 1):
        print(f"\n[{i}/{total}] Processing backup file: {backup_file}")
        if restore_database(backup_file):
            restore_count += 1
    
    print(f"\nRestoration complete. Successfully restored {restore_count} out of {total} databases.")

if __name__ == '__main__':
    main() 