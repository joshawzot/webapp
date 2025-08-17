#!/usr/bin/env python3
"""
Test improved storage detection logic
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from dual_storage_db_operations import DualStorageManager
from analyze_schema_ages import extract_timestamp_from_name
from db_operations import create_connection
from pathlib import Path
import json

def test_improved_detection():
    """Test the improved storage detection logic"""
    print("🧪 TESTING IMPROVED STORAGE DETECTION")
    print("=" * 60)
    
    try:
        # Create a fresh instance to force cache refresh
        dual_storage = DualStorageManager()
        
        # Get a sample of databases to test
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        all_databases = [db[0] for db in cursor.fetchall() 
                        if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        cursor.close()
        conn.close()
        
        # Test with a mix of databases
        test_databases = all_databases[:10]  # Test first 10
        
        print(f"📊 Testing {len(test_databases)} databases with improved detection:")
        print("-" * 80)
        
        # Check what's actually in the archive directory using ls command
        import subprocess
        try:
            result = subprocess.run(
                ['sudo', 'ls', '-1', '/local/mysql/data'],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                archive_schemas = result.stdout.strip().split('\n') if result.stdout.strip() else []
                print(f"📁 Found {len(archive_schemas)} schemas in archive directory")
            else:
                archive_schemas = []
                print("❌ Archive directory not found or empty")
        except Exception as e:
            archive_schemas = []
            print(f"⚠️  Could not check archive directory: {e}")
        
        # Check migration log
        migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
        migrated_schemas = set()
        if migration_log_path.exists():
            try:
                with open(migration_log_path, 'r') as f:
                    migration_log = json.load(f)
                
                for entry in migration_log:
                    if (entry.get('action') == 'VERIFY' and 
                        entry.get('status') == 'success'):
                        migrated_schemas.add(entry.get('schema'))
                
                print(f"📋 Migration log shows {len(migrated_schemas)} successfully migrated schemas")
            except Exception as e:
                print(f"⚠️  Could not read migration log: {e}")
        
        print("-" * 80)
        
        # Test storage detection for each database
        primary_count = 0
        secondary_count = 0
        
        for database in test_databases:
            location = dual_storage.get_schema_location(database)
            creation_time = extract_timestamp_from_name(database)
            
            # Check if it's in archive directory
            in_archive = database in archive_schemas
            in_migration_log = database in migrated_schemas
            
            if location == 'secondary':
                secondary_count += 1
                color = "🔵"
            else:
                primary_count += 1
                color = "🟢"
            
            age_str = ""
            if creation_time:
                from datetime import datetime
                age_days = (datetime.now() - creation_time).days
                age_str = f" ({age_days} days old)"
            
            print(f"{color} {database[:50]:<50} {location:<10} Archive:{in_archive} Log:{in_migration_log}{age_str}")
        
        print("-" * 80)
        print(f"📊 DETECTION SUMMARY:")
        print(f"   🟢 Primary Storage: {primary_count} schemas")
        print(f"   🔵 Archive Storage: {secondary_count} schemas")
        print(f"   📁 Total in archive dir: {len(archive_schemas)}")
        print(f"   📋 Total in migration log: {len(migrated_schemas)}")
        
        # Verify detection accuracy
        if secondary_count > 0:
            print(f"\n✅ SUCCESS: Found {secondary_count} schemas correctly identified as archived!")
        else:
            print(f"\n⚠️  WARNING: No schemas detected as archived. Checking detection logic...")
            
            # Debug: Check a known old schema
            old_schemas = [db for db in test_databases if extract_timestamp_from_name(db)]
            if old_schemas:
                test_schema = old_schemas[0]
                print(f"\n🔍 DEBUGGING with schema: {test_schema}")
                
                # Check if archive path exists using sudo ls
                try:
                    result = subprocess.run(
                        ['sudo', 'ls', '-d', f'/local/mysql/data/{test_schema}'],
                        capture_output=True, text=True, check=False
                    )
                    archive_exists = result.returncode == 0
                except:
                    archive_exists = False
                
                print(f"   Archive path exists: {archive_exists}")
                print(f"   In migration log: {test_schema in migrated_schemas}")
                
                creation_time = extract_timestamp_from_name(test_schema)
                if creation_time:
                    from datetime import datetime, timedelta
                    age_days = (datetime.now() - creation_time).days
                    is_old = creation_time < (datetime.now() - timedelta(days=60))
                    print(f"   Age: {age_days} days, Old (>60): {is_old}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_improved_detection()