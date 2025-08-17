#!/usr/bin/env python3
"""
Verify that the emergency fix successfully restored schema access
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import create_connection, fetch_tables
import json
from pathlib import Path

def test_schema_access(schema_name):
    """Test if schema has full data access."""
    try:
        conn = create_connection(schema_name)
        cursor = conn.cursor()
        
        # Get first table
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        if not tables:
            cursor.close()
            conn.close()
            return True, "Empty schema (no tables)"
        
        table_name = tables[0][0]
        
        # Try to access data - this was failing before
        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        return True, f"Full access: {len(tables)} tables, sample count: {count}"
        
    except Exception as e:
        return False, str(e)

def verify_fix_success():
    """Verify the emergency fix worked."""
    print("✅ VERIFYING EMERGENCY FIX SUCCESS")
    print("=" * 60)
    
    # Get some previously migrated schemas to test
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    test_schemas = []
    
    if migration_log_path.exists():
        with open(migration_log_path, 'r') as f:
            migration_log = json.load(f)
        
        for entry in migration_log:
            if (entry.get('action') == 'VERIFY' and 
                entry.get('status') == 'success'):
                test_schemas.append(entry.get('schema'))
                if len(test_schemas) >= 10:  # Test first 10
                    break
    
    print(f"🔍 Testing {len(test_schemas)} previously broken schemas:")
    print("-" * 60)
    
    working_count = 0
    broken_count = 0
    
    for i, schema_name in enumerate(test_schemas, 1):
        success, message = test_schema_access(schema_name)
        
        if success:
            status = "✅ WORKING"
            working_count += 1
        else:
            status = "❌ BROKEN"
            broken_count += 1
        
        print(f"{i:2d}. {schema_name[:50]:<50} {status}")
        print(f"    {message}")
    
    print("-" * 60)
    print(f"📊 RESULTS:")
    print(f"   ✅ Working schemas: {working_count}/{len(test_schemas)}")
    print(f"   ❌ Still broken: {broken_count}/{len(test_schemas)}")
    
    if broken_count == 0:
        print(f"\n🎉 SUCCESS! All tested schemas are now fully functional!")
        print(f"   • No more 'Tablespace is missing' errors")
        print(f"   • Full data access restored")
        print(f"   • Webapp should show normal table dimensions")
    elif broken_count < working_count:
        print(f"\n⚠️  PARTIAL SUCCESS: Most schemas fixed, {broken_count} still have issues")
    else:
        print(f"\n❌ ISSUES REMAIN: Emergency fix may not have worked fully")
    
    # Test webapp table display
    if working_count > 0:
        print(f"\n🌐 Testing webapp table display...")
        test_schema = test_schemas[0]
        try:
            tables = fetch_tables(test_schema)
            sample_table = tables[0] if tables else None
            
            if sample_table:
                dimensions = sample_table['dimensions']
                print(f"   📊 Sample table dimensions: {dimensions}")
                
                if 'archived' in dimensions:
                    print(f"   ⚠️  Still showing '(archived)' - may need webapp refresh")
                elif 'x' in dimensions and 'ERROR' not in dimensions:
                    print(f"   ✅ Normal dimensions displayed - webapp fix successful!")
                else:
                    print(f"   ❓ Unexpected format: {dimensions}")
            else:
                print(f"   📝 No tables in test schema")
                
        except Exception as e:
            print(f"   ❌ Webapp test failed: {e}")
    
    return working_count, broken_count

if __name__ == "__main__":
    verify_fix_success()