#!/usr/bin/env python3
"""
Test the improved webapp display for migrated schemas with tablespace issues
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import fetch_tables, create_connection
from dual_storage_db_operations import DualStorageManager
import json
from pathlib import Path

def test_webapp_display_fix():
    """Test the improved table dimension display for schemas with tablespace issues"""
    print("🧪 TESTING IMPROVED WEBAPP DISPLAY")
    print("=" * 70)
    
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
    
    if not migrated_schemas:
        print("❌ No migrated schemas found")
        return False
    
    # Test with first migrated schema
    test_schema = migrated_schemas[0]
    print(f"🔍 Testing table display for: {test_schema}")
    print("-" * 70)
    
    try:
        # Test the improved fetch_tables function
        tables = fetch_tables(test_schema)
        
        print(f"📊 Found {len(tables)} tables:")
        print(f"{'Table Name':<50} {'Dimensions':<20} {'Status'}")
        print("-" * 90)
        
        archived_count = 0
        error_count = 0
        working_count = 0
        
        for i, table in enumerate(tables[:10]):  # Show first 10 tables
            table_name = table['table_name']
            dimensions = table['dimensions']
            
            if 'archived' in dimensions:
                status = "📦 Archived"
                archived_count += 1
            elif 'error' in dimensions.lower():
                status = "❌ Error"
                error_count += 1
            else:
                status = "✅ Working"
                working_count += 1
            
            print(f"{table_name[:48]:<50} {dimensions:<20} {status}")
        
        if len(tables) > 10:
            print(f"... and {len(tables) - 10} more tables")
        
        print("-" * 90)
        print(f"📈 SUMMARY for {test_schema}:")
        print(f"   📦 Archived tables (tablespace issues): {archived_count}")
        print(f"   ✅ Working tables: {working_count}")
        print(f"   ❌ Error tables: {error_count}")
        
        # Test storage info display
        dual_storage = DualStorageManager()
        storage_location = dual_storage.get_schema_location(test_schema)
        
        print(f"\n💾 STORAGE INFO:")
        print(f"   Location: {storage_location}")
        print(f"   Expected in webapp: Archive Storage on /dev/sda1")
        
        if archived_count > 0:
            print(f"\n✅ SUCCESS: Tables now show '?x{tables[0]['dimensions'].split('x')[1] if 'x' in tables[0]['dimensions'] else 'N'} (archived)' instead of 'ERROR: Invalid view or table'")
        else:
            print(f"\n⚠️  No archived tables detected in this sample")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_webapp_display_fix()