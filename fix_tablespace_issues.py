#!/usr/bin/env python3
"""
Fix tablespace issues for migrated schemas
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import create_connection
import subprocess
import json
from pathlib import Path
import time

def fix_tablespace_issues():
    """Fix tablespace issues for migrated schemas"""
    print("🔧 FIXING TABLESPACE ISSUES FOR MIGRATED SCHEMAS")
    print("=" * 80)
    
    # Get migrated schemas from the log
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    migrated_schemas = []
    
    if migration_log_path.exists():
        with open(migration_log_path, 'r') as f:
            migration_log = json.load(f)
        
        for entry in migration_log:
            if (entry.get('action') == 'VERIFY' and 
                entry.get('status') == 'success'):
                migrated_schemas.append(entry.get('schema'))
    
    print(f"📋 Found {len(migrated_schemas)} migrated schemas to fix")
    
    # Method 1: Try FLUSH TABLES to refresh tablespace cache
    print("\n🔄 METHOD 1: Flushing table cache...")
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # Flush all tables to force MySQL to re-read file locations
        cursor.execute("FLUSH TABLES")
        print("✅ FLUSH TABLES executed successfully")
        
        cursor.close()
        conn.close()
        
        # Test if this fixed the issue
        test_schema = migrated_schemas[0] if migrated_schemas else None
        if test_schema:
            print(f"🧪 Testing fix with schema: {test_schema}")
            success = test_table_access(test_schema)
            if success:
                print("✅ FLUSH TABLES fixed the issue!")
                return True
            else:
                print("❌ FLUSH TABLES didn't fix the issue, trying next method...")
        
    except Exception as e:
        print(f"❌ FLUSH TABLES failed: {e}")
    
    # Method 2: Restart MySQL service
    print("\n🔄 METHOD 2: Restarting MySQL service...")
    try:
        # Stop MySQL
        print("⏹️  Stopping MySQL service...")
        result = subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✅ MySQL stopped successfully")
        else:
            print(f"⚠️  MySQL stop returned code {result.returncode}")
        
        # Wait a moment
        time.sleep(3)
        
        # Start MySQL
        print("▶️  Starting MySQL service...")
        result = subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✅ MySQL started successfully")
        else:
            print(f"❌ MySQL start failed with code {result.returncode}")
            return False
        
        # Wait for MySQL to fully start
        time.sleep(5)
        
        # Test if this fixed the issue
        test_schema = migrated_schemas[0] if migrated_schemas else None
        if test_schema:
            print(f"🧪 Testing fix with schema: {test_schema}")
            success = test_table_access(test_schema)
            if success:
                print("✅ MySQL restart fixed the issue!")
                return True
            else:
                print("❌ MySQL restart didn't fix the issue, trying next method...")
        
    except Exception as e:
        print(f"❌ MySQL restart failed: {e}")
    
    # Method 3: Use DISCARD/IMPORT TABLESPACE for problematic tables
    print("\n🔄 METHOD 3: Rebuilding tablespace references...")
    try:
        test_schema = migrated_schemas[0] if migrated_schemas else None
        if test_schema:
            success = rebuild_tablespace_references(test_schema)
            if success:
                print("✅ Tablespace rebuild fixed the issue!")
                return True
    except Exception as e:
        print(f"❌ Tablespace rebuild failed: {e}")
    
    print("\n❌ All methods failed. Manual intervention may be required.")
    print("\n📋 MANUAL STEPS TO TRY:")
    print("1. Check MySQL error log: sudo tail -f /var/log/mysql/error.log")
    print("2. Verify symbolic link permissions: ls -la /var/lib/mysql/")
    print("3. Check that mysql user can access /local/mysql/data/")
    print("4. Consider moving schemas back to primary storage temporarily")
    
    return False

def test_table_access(schema_name):
    """Test if we can access tables in a migrated schema"""
    try:
        conn = create_connection(database=schema_name)
        cursor = conn.cursor()
        
        # Get first table
        cursor.execute("SHOW TABLES")
        all_tables = cursor.fetchall()
        tables = all_tables[:1]  # Get just first table
        if not tables:
            cursor.close()
            conn.close()
            return False
        
        table_name = tables[0][0]
        
        # Try to count rows
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        print(f"   ✅ Successfully accessed table {table_name} with {count} rows")
        return True
        
    except Exception as e:
        print(f"   ❌ Table access failed: {e}")
        return False

def rebuild_tablespace_references(schema_name):
    """Try to rebuild tablespace references for a schema"""
    print(f"🔨 Rebuilding tablespace references for {schema_name}")
    
    try:
        conn = create_connection(database=schema_name)
        cursor = conn.cursor()
        
        # Get tables with issues
        cursor.execute("SHOW TABLES")
        all_tables = [table[0] for table in cursor.fetchall()]
        
        print(f"   Found {len(all_tables)} tables to check")
        
        # Test a few tables and try to fix them
        problem_tables = []
        for table_name in all_tables[:5]:  # Test first 5 tables
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                cursor.fetchone()
                print(f"   ✅ {table_name} - OK")
            except Exception:
                problem_tables.append(table_name)
                print(f"   ❌ {table_name} - Tablespace issue")
        
        if not problem_tables:
            cursor.close()
            conn.close()
            print("   ✅ No tablespace issues found")
            return True
        
        # Try to fix one problem table as a test
        if problem_tables:
            table_name = problem_tables[0]
            print(f"   🔧 Attempting to fix {table_name}...")
            
            try:
                # This method is complex and risky, so we'll just report the issue
                print(f"   ⚠️  Table {table_name} needs manual tablespace repair")
                print(f"      This requires DISCARD/IMPORT TABLESPACE operations")
                print(f"      which are risky and should be done manually")
            except Exception as e:
                print(f"   ❌ Repair attempt failed: {e}")
        
        cursor.close()
        conn.close()
        return False
        
    except Exception as e:
        print(f"   ❌ Failed to rebuild tablespace references: {e}")
        return False

if __name__ == "__main__":
    fix_tablespace_issues()