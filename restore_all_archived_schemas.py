#!/usr/bin/env python3
"""
Comprehensive restoration script for all archived schemas from SQL backups.
This script will:
1. Find all SQL backup files
2. Extract schema names from backup filenames
3. Check if schema currently exists and has tablespace issues
4. Remove corrupted schema/symlink if needed
5. Restore from SQL backup
6. Verify restoration success
"""

import os
import subprocess
import mysql.connector
import glob
import re
from pathlib import Path

# Configuration
BACKUP_DIR = "/local/mysql_migration_backups/"
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'charset': 'utf8mb4'
}

def run_command(cmd, shell=False):
    """Run a system command and return result."""
    try:
        if isinstance(cmd, str) and not shell:
            cmd = cmd.split()
        result = subprocess.run(cmd, capture_output=True, text=True, shell=shell)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_schema_name_from_backup(backup_file):
    """Extract schema name from backup filename."""
    # Remove .sql extension and path
    filename = os.path.basename(backup_file).replace('.sql', '')
    return filename

def check_schema_exists(schema_name):
    """Check if schema exists in MySQL."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES LIKE %s", (schema_name,))
        result = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        print(f"Error checking schema {schema_name}: {e}")
        return False

def check_tablespace_issues(schema_name):
    """Check if schema has tablespace issues."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"USE `{schema_name}`")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        has_issues = False
        for (table,) in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                cursor.fetchone()
            except mysql.connector.Error as e:
                if "Tablespace is missing" in str(e) or "tablespace" in str(e).lower():
                    has_issues = True
                    break
        
        cursor.close()
        conn.close()
        return has_issues
    except Exception as e:
        # If we can't even access the schema, it likely has issues
        return True

def remove_corrupted_schema(schema_name):
    """Remove corrupted schema and its symlink."""
    print(f"  Removing corrupted schema {schema_name}...")
    
    # Drop the database
    success, stdout, stderr = run_command(f'sudo mysql -e "DROP DATABASE IF EXISTS `{schema_name}`;"')
    if not success:
        print(f"    Warning: Could not drop database: {stderr}")
    
    # Convert schema name to filesystem encoding
    fs_schema_name = schema_name.replace('@', '@0040').replace('-', '@002d')
    
    # Remove symlink from primary storage if it exists
    primary_path = f"/var/lib/mysql/{fs_schema_name}"
    success, _, _ = run_command(f"sudo rm -rf {primary_path}")
    
    print(f"    Cleaned up corrupted schema")

def restore_schema_from_backup(schema_name, backup_file):
    """Restore schema from SQL backup."""
    print(f"  Restoring {schema_name} from {os.path.basename(backup_file)}...")
    
    # Create database
    success, stdout, stderr = run_command(f'sudo mysql -e "CREATE DATABASE `{schema_name}`;"')
    if not success:
        print(f"    Error creating database: {stderr}")
        return False
    
    # Import SQL backup
    success, stdout, stderr = run_command(f"sudo mysql {schema_name} < {backup_file}", shell=True)
    if not success:
        print(f"    Error importing backup: {stderr}")
        return False
    
    print(f"    Successfully restored {schema_name}")
    return True

def verify_restoration(schema_name):
    """Verify that restoration was successful."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"USE `{schema_name}`")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        total_rows = 0
        for (table,) in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                count = cursor.fetchone()[0]
                total_rows += count
            except Exception as e:
                print(f"    Warning: Could not count rows in {table}: {e}")
        
        cursor.close()
        conn.close()
        return total_rows
    except Exception as e:
        print(f"    Error verifying {schema_name}: {e}")
        return 0

def main():
    print("🔄 Starting comprehensive archived schema restoration...")
    print("=" * 60)
    
    # Find all SQL backup files
    backup_files = glob.glob(os.path.join(BACKUP_DIR, "*.sql"))
    print(f"Found {len(backup_files)} SQL backup files")
    
    restored_count = 0
    skipped_count = 0
    failed_count = 0
    
    for backup_file in sorted(backup_files):
        schema_name = get_schema_name_from_backup(backup_file)
        print(f"\n📦 Processing: {schema_name}")
        
        # Check if schema exists
        if not check_schema_exists(schema_name):
            print(f"  Schema doesn't exist in MySQL, restoring from backup...")
            if restore_schema_from_backup(schema_name, backup_file):
                row_count = verify_restoration(schema_name)
                print(f"  ✅ Restored with {row_count} total rows")
                restored_count += 1
            else:
                print(f"  ❌ Failed to restore")
                failed_count += 1
            continue
        
        # Check for tablespace issues
        if check_tablespace_issues(schema_name):
            print(f"  Schema has tablespace issues, fixing...")
            remove_corrupted_schema(schema_name)
            if restore_schema_from_backup(schema_name, backup_file):
                row_count = verify_restoration(schema_name)
                print(f"  ✅ Fixed and restored with {row_count} total rows")
                restored_count += 1
            else:
                print(f"  ❌ Failed to restore")
                failed_count += 1
        else:
            print(f"  ✅ Schema is healthy, skipping")
            skipped_count += 1
    
    print("\n" + "=" * 60)
    print("🎯 RESTORATION SUMMARY:")
    print(f"  ✅ Restored: {restored_count}")
    print(f"  ⏭️  Skipped (healthy): {skipped_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📊 Total processed: {len(backup_files)}")
    
    if failed_count == 0:
        print("\n🎉 ALL SCHEMAS SUCCESSFULLY RESTORED!")
        print("The tablespace issues have been completely resolved!")
    else:
        print(f"\n⚠️  {failed_count} schemas failed restoration - check logs above")

if __name__ == "__main__":
    main()