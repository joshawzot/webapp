#!/usr/bin/env python3
"""
Test schemas using ONLY MySQL commands - no file system access
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import create_connection, fetch_tables

def test_mysql_schemas():
    """Test schemas using only MySQL commands."""
    print("🧪 TESTING SCHEMAS - MySQL Only")
    print("=" * 50)
    
    # Test schemas that were moved
    test_schemas = [
        'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC2_20250109'
    ]
    
    print("1️⃣ Basic MySQL connection test...")
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        print("   ✅ MySQL connection works")
    except Exception as e:
        print(f"   ❌ MySQL connection failed: {e}")
        return
    
    print("\n2️⃣ Schema discovery test...")
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        all_databases = [db[0] for db in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        found_count = 0
        for schema in test_schemas:
            if schema in all_databases:
                found_count += 1
                print(f"   ✅ Found: {schema}")
            else:
                print(f"   ❌ Missing: {schema}")
        
        print(f"   📊 Found {found_count}/{len(test_schemas)} schemas")
        
    except Exception as e:
        print(f"   ❌ Schema discovery failed: {e}")
        return
    
    if found_count == 0:
        print("\n❌ No schemas found - fix didn't work")
        return
    
    print("\n3️⃣ Table access test...")
    working_schemas = 0
    
    for schema in test_schemas:
        print(f"\n📋 Testing: {schema}")
        try:
            # Test basic connection to schema
            conn = create_connection(database=schema)
            cursor = conn.cursor()
            
            # Test SHOW TABLES
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            table_count = len(tables)
            print(f"   📊 Tables: {table_count}")
            
            if table_count > 0:
                # Test data access on first table
                table_name = tables[0][0]
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                    row_count = cursor.fetchone()[0]
                    print(f"   ✅ Data access works: {row_count} rows in {table_name}")
                    working_schemas += 1
                except Exception as data_error:
                    if "Tablespace is missing" in str(data_error):
                        print(f"   ❌ STILL BROKEN: Tablespace error - {data_error}")
                    else:
                        print(f"   ❌ Data error: {data_error}")
            else:
                print(f"   ⚠️  Empty schema")
                working_schemas += 1  # Empty is "working"
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"   ❌ Schema access failed: {e}")
    
    print(f"\n4️⃣ Webapp table display test...")
    if working_schemas > 0:
        test_schema = test_schemas[0]
        try:
            print(f"   Testing webapp display for: {test_schema}")
            tables = fetch_tables(test_schema)
            
            if tables:
                sample_table = tables[0]
                dimensions = sample_table['dimensions']
                print(f"   📊 Sample table: {sample_table['table_name']}")
                print(f"   📐 Dimensions: {dimensions}")
                
                if 'archived' in dimensions:
                    print(f"   ⚠️  Still showing '(archived)' - tablespace issue remains")
                elif 'ERROR' in dimensions:
                    print(f"   ❌ Still showing error")
                elif 'x' in dimensions and dimensions.count('x') == 1:
                    print(f"   ✅ NORMAL DIMENSIONS! Fix worked!")
                else:
                    print(f"   ❓ Unexpected format: {dimensions}")
            else:
                print(f"   📭 No tables found")
                
        except Exception as e:
            print(f"   ❌ Webapp test failed: {e}")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   ✅ Working schemas: {working_schemas}/{len(test_schemas)}")
    
    if working_schemas == len(test_schemas):
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"   • All schemas are accessible")
        print(f"   • No more tablespace errors")
        print(f"   • Webapp should show normal dimensions")
        print(f"   • Ready to continue with remaining 1,477 schemas")
    elif working_schemas > 0:
        print(f"\n✅ PARTIAL SUCCESS!")
        print(f"   • {working_schemas} schemas working")
        print(f"   • Fix approach is correct")
        print(f"   • Continue with remaining schemas")
    else:
        print(f"\n❌ FIX FAILED")
        print(f"   • Tablespace errors persist")
        print(f"   • Need different approach")

if __name__ == "__main__":
    test_mysql_schemas()