#!/usr/bin/env python3
"""
Test the fixed storage detection logic
Should now show schemas as primary storage after nuclear fix
"""

import sys
sys.path.append('/home/admin2/webapp_2')
from dual_storage_db_operations import DualStorageManager

def main():
    print("🔍 TESTING FIXED STORAGE DETECTION")
    print("=" * 60)
    print("After nuclear fix, all schemas should show as primary storage")
    print()
    
    # Create storage manager (will force cache refresh in __init__)
    dual_storage = DualStorageManager()
    
    # Test the specific schema the user mentioned
    test_schema = "MaxZhang_Cullinan_2331_JH1_I03_KPI_20250327082452"
    
    print(f"🎯 Testing specific schema: {test_schema}")
    
    # Get location info
    location_info = dual_storage.get_schema_location(test_schema)
    
    if location_info:
        print(f"   📁 Storage Location: {location_info['storage_location']}")
        print(f"   💾 Device: {location_info['device']}")
        print(f"   📂 Path: {location_info['path']}")
        print(f"   📅 Age: {location_info['age_days']} days old")
        print(f"   🕐 Created: {location_info['creation_date']}")
        
        if location_info['storage_location'] == 'Primary Storage':
            print(f"   ✅ CORRECT: Shows as Primary Storage")
        else:
            print(f"   ❌ WRONG: Still shows as {location_info['storage_location']}")
    else:
        print(f"   ❌ Schema not found or error getting location")
    
    # Test a few more schemas
    print(f"\n🔍 Testing additional schemas:")
    
    # Get a few more schemas to test
    import mysql.connector
    from db_operations import create_connection
    
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        all_dbs = cursor.fetchall()
        maxzhang_dbs = [db[0] for db in all_dbs if 'MaxZhang' in db[0]][:5]  # Test first 5
        cursor.close()
        conn.close()
        
        primary_count = 0
        secondary_count = 0
        
        for schema in maxzhang_dbs:
            location_info = dual_storage.get_schema_location(schema)
            if location_info:
                storage_type = location_info['storage_location']
                if storage_type == 'Primary Storage':
                    primary_count += 1
                    print(f"   ✅ {schema}: Primary Storage")
                else:
                    secondary_count += 1
                    print(f"   ⚠️  {schema}: {storage_type}")
            else:
                print(f"   ❌ {schema}: Error getting location")
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   ✅ Primary Storage: {primary_count}/{len(maxzhang_dbs)}")
        print(f"   📦 Archive Storage: {secondary_count}/{len(maxzhang_dbs)}")
        
        if primary_count == len(maxzhang_dbs):
            print(f"\n🎉 PERFECT! All schemas correctly show as Primary Storage")
            print(f"   • Storage detection fixed")
            print(f"   • Webapp should now show correct storage info")
            print(f"   • Refresh your webapp page to see the fix")
        elif primary_count > secondary_count:
            print(f"\n🎉 MOSTLY FIXED! Most schemas show as Primary Storage")
            print(f"   • {secondary_count} schemas may still have cache issues")
            print(f"   • Try refreshing webapp or wait a few minutes")
        else:
            print(f"\n❌ STILL ISSUES: Most schemas show as Archive Storage")
            print(f"   • Storage detection may need additional fixes")
            print(f"   • Check file permissions or cache logic")
        
    except Exception as e:
        print(f"❌ Error testing schemas: {e}")

if __name__ == "__main__":
    main()