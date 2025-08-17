#!/usr/bin/env python3
"""
Diagnose tablespace issues with migrated schemas
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import create_connection
from dual_storage_db_operations import DualStorageManager
import subprocess
import json
from pathlib import Path

def diagnose_tablespace_issues():
    """Diagnose why migrated schemas have tablespace issues"""
    print("🔍 DIAGNOSING TABLESPACE ISSUES")
    print("=" * 80)
    
    # Get a migrated schema from the log
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    migrated_schemas = []
    
    if migration_log_path.exists():
        with open(migration_log_path, 'r') as f:
            migration_log = json.load(f)
        
        for entry in migration_log:
            if (entry.get('action') == 'VERIFY' and 
                entry.get('status') == 'success'):
                migrated_schemas.append(entry.get('schema'))
    
    # Test with first migrated schema
    if not migrated_schemas:
        print("❌ No migrated schemas found in log")
        return
    
    test_schema = migrated_schemas[0]
    print(f"🧪 Testing migrated schema: {test_schema}")
    print("-" * 80)
    
    try:
        # Connect to MySQL and try to access the schema
        conn = create_connection(database=test_schema)
        cursor = conn.cursor()
        
        print("✅ Successfully connected to migrated schema")
        
        # Try to show tables
        print("📋 Checking tables in schema...")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   Found {len(tables)} tables")
        
        if tables:
            # Test first table
            table_name = tables[0][0]
            print(f"\n🔍 Testing table: {table_name}")
            
            # Try to describe the table
            try:
                cursor.execute(f"DESCRIBE `{table_name}`")
                columns = cursor.fetchall()
                print(f"   ✅ DESCRIBE successful - {len(columns)} columns")
            except Exception as e:
                print(f"   ❌ DESCRIBE failed: {e}")
            
            # Try to count rows
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                print(f"   ✅ COUNT successful - {count} rows")
            except Exception as e:
                print(f"   ❌ COUNT failed: {e}")
                
            # Try to get table status
            try:
                cursor.execute(f"SHOW TABLE STATUS LIKE '{table_name}'")
                status = cursor.fetchone()
                if status:
                    print(f"   ✅ TABLE STATUS successful")
                    print(f"      Engine: {status[1]}")
                    print(f"      Rows: {status[4]}")
                    print(f"      Data Length: {status[6]}")
                else:
                    print(f"   ❌ TABLE STATUS returned no results")
            except Exception as e:
                print(f"   ❌ TABLE STATUS failed: {e}")
                
            # Check tablespace information
            try:
                cursor.execute(f"""
                    SELECT TABLE_NAME, ENGINE, TABLE_ROWS, DATA_LENGTH
                    FROM information_schema.TABLES 
                    WHERE TABLE_SCHEMA = '{test_schema}' 
                    AND TABLE_NAME = '{table_name}'
                """)
                info = cursor.fetchone()
                if info:
                    print(f"   ✅ INFORMATION_SCHEMA access successful")
                    print(f"      Engine: {info[1]}, Rows: {info[2]}, Size: {info[3]} bytes")
                else:
                    print(f"   ❌ INFORMATION_SCHEMA returned no results")
            except Exception as e:
                print(f"   ❌ INFORMATION_SCHEMA query failed: {e}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Failed to connect to schema: {e}")
    
    print("\n" + "=" * 80)
    print("🔍 CHECKING FILE SYSTEM STRUCTURE")
    print("-" * 80)
    
    # Check symbolic link structure
    try:
        primary_path = f"/var/lib/mysql/{test_schema}"
        result = subprocess.run(['sudo', 'ls', '-la', primary_path], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"📁 Primary path ({primary_path}):")
            print(f"   {result.stdout.strip()}")
        else:
            print(f"❌ Primary path not found: {primary_path}")
    except Exception as e:
        print(f"❌ Error checking primary path: {e}")
    
    # Check archive path  
    try:
        archive_path = f"/local/mysql/data/{test_schema}"
        result = subprocess.run(['sudo', 'ls', '-la', archive_path], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"\n📁 Archive path ({archive_path}):")
            # Show first few files
            lines = result.stdout.strip().split('\n')[:10]
            for line in lines:
                print(f"   {line}")
            total_lines = len(result.stdout.strip().split('\n'))
            if total_lines > 10:
                print(f"   ... and {total_lines - 10} more files")
        else:
            print(f"❌ Archive path not found: {archive_path}")
    except Exception as e:
        print(f"❌ Error checking archive path: {e}")
    
    print("\n" + "=" * 80)
    print("🔍 CHECKING MYSQL CONFIGURATION")
    print("-" * 80)
    
    # Check MySQL data directory and innodb settings
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # Check datadir
        cursor.execute("SELECT @@datadir")
        datadir = cursor.fetchone()[0]
        print(f"📂 MySQL datadir: {datadir}")
        
        # Check innodb settings
        cursor.execute("SHOW VARIABLES LIKE 'innodb_file_per_table'")
        result = cursor.fetchone()
        print(f"🗃️ innodb_file_per_table: {result[1] if result else 'Unknown'}")
        
        cursor.execute("SHOW VARIABLES LIKE 'innodb_data_home_dir'")
        result = cursor.fetchone()
        print(f"🏠 innodb_data_home_dir: {result[1] if result else 'Unknown'}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking MySQL configuration: {e}")
    
    print("\n" + "=" * 80)
    print("📋 RECOMMENDATIONS:")
    print("1. Check if symbolic links are properly created")
    print("2. Verify file permissions on archive directory")
    print("3. Consider using MySQL's tablespace import/export instead of file copying")
    print("4. May need to restart MySQL service after migration")
    print("5. Check if innodb_file_per_table is enabled")

if __name__ == "__main__":
    diagnose_tablespace_issues()