#!/usr/bin/env python3
"""
Targeted fix for schemas with tablespace issues.
This script focuses only on schemas that currently exist and have tablespace problems.
"""

import os
import subprocess
import mysql.connector
import glob

# Configuration
BACKUP_DIR = "/local/mysql_migration_backups/"
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'charset': 'utf8mb4'
}

def run_mysql_command(cmd):
    """Run a MySQL command and return result."""
    try:
        result = subprocess.run(['sudo', 'mysql', '-e', cmd], 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_all_schemas():
    """Get list of all current schemas."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        schemas = [row[0] for row in cursor.fetchall() 
                  if row[0] not in ['information_schema', 'performance_schema', 'mysql', 'sys']]
        cursor.close()
        conn.close()
        return schemas
    except Exception as e:
        print(f"Error getting schemas: {e}")
        return []

def check_tablespace_issues(schema_name):
    """Check if a schema has tablespace issues."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG, database=schema_name)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        issues = []
        for (table,) in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}` LIMIT 1")
                cursor.fetchone()
            except mysql.connector.Error as e:
                if "Tablespace is missing" in str(e) or "tablespace" in str(e).lower():
                    issues.append((table, str(e)))
        
        cursor.close()
        conn.close()
        return issues
    except Exception as e:
        return [("SCHEMA_ACCESS", str(e))]

def find_backup_file(schema_name):
    """Find the backup file for a schema."""
    backup_pattern = os.path.join(BACKUP_DIR, f"{schema_name}.sql")
    if os.path.exists(backup_pattern):
        return backup_pattern
    return None

def restore_schema(schema_name, backup_file):
    """Restore schema from backup."""
    print(f"  🔄 Restoring {schema_name}...")
    
    # Drop existing database
    success, stdout, stderr = run_mysql_command(f"DROP DATABASE IF EXISTS `{schema_name}`")
    if not success:
        print(f"    ⚠️ Warning dropping database: {stderr}")
    
    # Create new database
    success, stdout, stderr = run_mysql_command(f"CREATE DATABASE `{schema_name}`")
    if not success:
        print(f"    ❌ Failed to create database: {stderr}")
        return False
    
    # Import backup
    try:
        result = subprocess.run(['sudo', 'mysql', schema_name], 
                              input=open(backup_file, 'r').read(),
                              text=True, capture_output=True)
        if result.returncode != 0:
            print(f"    ❌ Failed to import backup: {result.stderr}")
            return False
    except Exception as e:
        print(f"    ❌ Exception during import: {e}")
        return False
    
    print(f"    ✅ Successfully restored")
    return True

def verify_fix(schema_name):
    """Verify that the fix worked."""
    issues = check_tablespace_issues(schema_name)
    if not issues:
        # Count total rows to verify data
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG, database=schema_name)
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            total_rows = 0
            for (table,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                count = cursor.fetchone()[0]
                total_rows += count
            
            cursor.close()
            conn.close()
            return total_rows
        except:
            return 0
    return None

def main():
    print("🎯 Targeted Tablespace Issue Fix")
    print("=" * 50)
    
    # Get all current schemas
    schemas = get_all_schemas()
    print(f"Found {len(schemas)} schemas to check")
    
    problem_schemas = []
    
    # Check each schema for tablespace issues
    print("\n🔍 Scanning for tablespace issues...")
    for schema in schemas:
        issues = check_tablespace_issues(schema)
        if issues:
            print(f"  ❌ {schema}: {len(issues)} table(s) with issues")
            problem_schemas.append((schema, issues))
        else:
            print(f"  ✅ {schema}: Healthy")
    
    if not problem_schemas:
        print("\n🎉 No tablespace issues found!")
        return
    
    print(f"\n🔧 Found {len(problem_schemas)} schemas with issues")
    print("=" * 50)
    
    fixed_count = 0
    failed_count = 0
    
    for schema_name, issues in problem_schemas:
        print(f"\n📦 Fixing: {schema_name}")
        print(f"   Issues: {[issue[0] for issue in issues]}")
        
        # Find backup file
        backup_file = find_backup_file(schema_name)
        if not backup_file:
            print(f"    ❌ No backup file found for {schema_name}")
            failed_count += 1
            continue
        
        # Restore from backup
        if restore_schema(schema_name, backup_file):
            row_count = verify_fix(schema_name)
            if row_count is not None:
                print(f"    ✅ Fixed! Schema has {row_count} total rows")
                fixed_count += 1
            else:
                print(f"    ⚠️ Restored but still has issues")
                failed_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY:")
    print(f"  ✅ Fixed: {fixed_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📊 Total problems: {len(problem_schemas)}")
    
    if failed_count == 0:
        print("\n🎉 ALL TABLESPACE ISSUES RESOLVED!")

if __name__ == "__main__":
    main()