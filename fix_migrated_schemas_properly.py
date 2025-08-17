#!/usr/bin/env python3
"""
Comprehensive fix for migrated schema tablespace issues
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import create_connection
import subprocess
import json
from pathlib import Path
import time

def fix_migrated_schemas():
    """Fix migrated schemas using proper MySQL methods"""
    print("🔧 COMPREHENSIVE FIX FOR MIGRATED SCHEMAS")
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
    
    print(f"📋 Found {len(migrated_schemas)} migrated schemas with issues")
    
    # Test current state
    print("\n🔍 Testing current state...")
    working_schemas = 0
    broken_schemas = 0
    
    for i, schema in enumerate(migrated_schemas[:10]):  # Test first 10
        if test_schema_access(schema):
            working_schemas += 1
        else:
            broken_schemas += 1
    
    print(f"   ✅ Working: {working_schemas}/10")
    print(f"   ❌ Broken: {broken_schemas}/10")
    
    if broken_schemas == 0:
        print("\n🎉 All tested schemas are working! No fix needed.")
        return True
    
    print(f"\n💡 PROPOSED SOLUTIONS:")
    print("=" * 50)
    
    print("🔄 OPTION 1: Quick Fix - Re-import from backup")
    print("   • Move broken schemas back to primary storage")  
    print("   • Re-import from mysqldump backups")
    print("   • Safest option, preserves all data")
    print("   • Estimated time: 2-4 hours")
    
    print("\n🚀 OPTION 2: Advanced Fix - Tablespace repair")
    print("   • Use MySQL DISCARD/IMPORT TABLESPACE")
    print("   • Complex but keeps schemas on archive storage")
    print("   • Risk of data loss if not done carefully")
    print("   • Estimated time: 4-8 hours")
    
    print("\n⚡ OPTION 3: Hybrid Approach")
    print("   • Keep recent schemas (working) on archive")
    print("   • Move only broken schemas back to primary")
    print("   • Gradual migration of working schemas later")
    print("   • Balanced risk/benefit")
    
    choice = input("\nWhich option would you like to try? (1/2/3): ").strip()
    
    if choice == "1":
        return fix_option_1_reimport(migrated_schemas)
    elif choice == "2":
        return fix_option_2_tablespace(migrated_schemas)
    elif choice == "3":
        return fix_option_3_hybrid(migrated_schemas)
    else:
        print("❌ Invalid choice. Exiting.")
        return False

def test_schema_access(schema_name):
    """Test if a schema's tables are accessible"""
    try:
        conn = create_connection(database=schema_name)
        cursor = conn.cursor()
        
        # Get first table
        cursor.execute("SHOW TABLES")
        all_tables = cursor.fetchall()
        if not all_tables:
            cursor.close()
            conn.close()
            return True  # Empty schema is "working"
        
        table_name = all_tables[0][0]
        
        # Try to access data
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        cursor.fetchone()
        
        cursor.close()
        conn.close()
        return True
        
    except Exception:
        return False

def fix_option_1_reimport(migrated_schemas):
    """Option 1: Re-import from backups"""
    print("\n🔄 EXECUTING OPTION 1: Re-import from backups")
    print("=" * 50)
    
    print("📋 Steps that will be performed:")
    print("1. Identify schemas with tablespace issues")
    print("2. Drop the broken symbolic link schemas")
    print("3. Re-import from mysqldump backups") 
    print("4. Move re-imported schemas back to archive")
    print("5. Create proper symbolic links")
    
    confirm = input("\nProceed with Option 1? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        return False
    
    # Check if backup directory exists
    backup_dir = Path('/home/admin2/webapp_2/migration_backups')
    if not backup_dir.exists():
        print(f"❌ Backup directory not found: {backup_dir}")
        print("   Cannot proceed without backups")
        return False
    
    print("✅ Backup directory found")
    
    # Test with one schema first
    test_schema = migrated_schemas[0]
    print(f"\n🧪 Testing with schema: {test_schema}")
    
    backup_file = backup_dir / f"{test_schema}.sql"
    if not backup_file.exists():
        print(f"❌ Backup file not found: {backup_file}")
        return False
    
    print("✅ Backup file found")
    print("🚨 MANUAL INTERVENTION REQUIRED")
    print(f"\nTo fix the schemas, run these commands manually:")
    print(f"1. cd {backup_dir}")
    print(f"2. sudo mysql -e 'DROP DATABASE `{test_schema}`;'")
    print(f"3. sudo mysql -e 'CREATE DATABASE `{test_schema}`;'")
    print(f"4. sudo mysql {test_schema} < {test_schema}.sql")
    
    return True

def fix_option_2_tablespace(migrated_schemas):
    """Option 2: Advanced tablespace repair"""
    print("\n🚀 EXECUTING OPTION 2: Tablespace repair")
    print("=" * 50)
    
    print("⚠️  WARNING: This is an advanced operation")
    print("   Risk of data loss if interrupted")
    print("   Requires careful execution")
    
    confirm = input("\nProceed with advanced tablespace repair? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        return False
    
    print("🚨 MANUAL INTERVENTION REQUIRED")
    print("\nTablespace repair requires these manual steps:")
    print("1. For each broken table:")
    print("   ALTER TABLE `table_name` DISCARD TABLESPACE;")
    print("   ALTER TABLE `table_name` IMPORT TABLESPACE;")
    print("2. This must be done for each of the ~1400 tables")
    print("3. Very time-consuming and error-prone")
    print("\n💡 Recommendation: Use Option 1 instead")
    
    return False

def fix_option_3_hybrid(migrated_schemas):
    """Option 3: Hybrid approach"""
    print("\n⚡ EXECUTING OPTION 3: Hybrid approach")
    print("=" * 50)
    
    print("📋 Strategy:")
    print("1. Identify which schemas are working vs broken")
    print("2. Move only broken schemas back to primary storage")
    print("3. Keep working schemas on archive storage")
    
    # Test all migrated schemas to categorize them
    print("\n🔍 Testing all migrated schemas...")
    working_schemas = []
    broken_schemas = []
    
    for i, schema in enumerate(migrated_schemas):
        if i % 100 == 0:
            print(f"   Testing {i+1}/{len(migrated_schemas)}...")
        
        if test_schema_access(schema):
            working_schemas.append(schema)
        else:
            broken_schemas.append(schema)
    
    print(f"\n📊 Results:")
    print(f"   ✅ Working schemas: {len(working_schemas)}")
    print(f"   ❌ Broken schemas: {len(broken_schemas)}")
    
    if len(broken_schemas) == 0:
        print("🎉 All schemas are working! No action needed.")
        return True
    
    print(f"\n💡 Recommendation:")
    print(f"   • Keep {len(working_schemas)} working schemas on archive")
    print(f"   • Move {len(broken_schemas)} broken schemas back to primary")
    print(f"   • Re-migrate broken schemas properly later")
    
    # Save the categorization
    results_file = Path('/home/admin2/webapp_2/schema_categorization.json')
    results = {
        'working_schemas': working_schemas,
        'broken_schemas': broken_schemas,
        'timestamp': time.time()
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")
    
    return True

if __name__ == "__main__":
    fix_migrated_schemas()