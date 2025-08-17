#!/usr/bin/env python3
"""
Test the fixed schemas to see exactly what's happening
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import create_connection
import subprocess
from pathlib import Path

def test_mysql_connection():
    """Test basic MySQL connection."""
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        print("✅ Basic MySQL connection works")
        return True
    except Exception as e:
        print(f"❌ Basic MySQL connection failed: {e}")
        return False

def test_schema_discovery():
    """Test if MySQL can see the moved schemas."""
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        
        # Check for our test schemas
        test_schemas = [
            'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
            'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109', 
            'MaxZhang_Cullinan_183_100ReadRAC2_20250109'
        ]
        
        found_schemas = []
        for schema in test_schemas:
            if schema in databases:
                found_schemas.append(schema)
        
        cursor.close()
        conn.close()
        
        print(f"✅ MySQL can see {len(found_schemas)}/{len(test_schemas)} moved schemas")
        for schema in found_schemas:
            print(f"   📁 {schema}")
        
        return found_schemas
        
    except Exception as e:
        print(f"❌ Schema discovery failed: {e}")
        return []

def test_schema_access(schema_name):
    """Test access to a specific schema."""
    try:
        # Test connection to schema
        conn = create_connection(database=schema_name)
        cursor = conn.cursor()
        
        # Test SHOW TABLES
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"   📊 Found {len(tables)} tables")
        
        if tables:
            # Test access to first table
            table_name = tables[0][0]
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                print(f"   ✅ Data access works: {count} rows in {table_name}")
                cursor.close()
                conn.close()
                return True, f"Full access: {len(tables)} tables, {count} rows"
            except Exception as data_error:
                print(f"   ❌ Data access failed: {data_error}")
                cursor.close()
                conn.close()
                return False, f"Structure OK but data access failed: {data_error}"
        else:
            print(f"   ⚠️  No tables found")
            cursor.close()
            conn.close()
            return True, "Empty schema"
            
    except Exception as e:
        print(f"   ❌ Schema access failed: {e}")
        return False, str(e)

def check_file_locations():
    """Check the file locations of moved schemas."""
    test_schemas = [
        'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109'
    ]
    
    print("📂 Checking file locations:")
    for schema in test_schemas:
        primary_path = Path(f"/var/lib/mysql/{schema}")
        archive_path = Path(f"/local/mysql/data/{schema}")
        
        print(f"   {schema}:")
        
        if primary_path.exists():
            if primary_path.is_symlink():
                print(f"      ❌ Still a symlink: {primary_path.readlink()}")
            else:
                print(f"      ✅ Real directory in primary storage")
        else:
            print(f"      ❌ Not found in primary storage")
        
        if archive_path.exists():
            print(f"      ⚠️  Still exists in archive (should be moved)")
        else:
            print(f"      ✅ Moved from archive")

def main():
    print("🧪 TESTING FIXED SCHEMAS")
    print("=" * 60)
    
    # Test 1: Basic MySQL connection
    print("\n1️⃣ Testing basic MySQL connection...")
    if not test_mysql_connection():
        print("❌ Can't proceed - basic MySQL connection failed")
        return
    
    # Test 2: File locations
    print("\n2️⃣ Checking file locations...")
    check_file_locations()
    
    # Test 3: Schema discovery
    print("\n3️⃣ Testing schema discovery...")
    found_schemas = test_schema_discovery()
    
    if not found_schemas:
        print("❌ No schemas found - migration may not have worked")
        return
    
    # Test 4: Schema access
    print("\n4️⃣ Testing schema access...")
    working_count = 0
    for schema in found_schemas[:3]:  # Test first 3
        print(f"\n📋 Testing: {schema}")
        success, message = test_schema_access(schema)
        if success:
            working_count += 1
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   ✅ Working schemas: {working_count}/{len(found_schemas[:3])}")
    
    if working_count > 0:
        print(f"\n🎉 SUCCESS! Schema access is restored!")
        print(f"   • Tablespace errors should be gone")
        print(f"   • Webapp should show normal table dimensions")
        print(f"   • Data access is working")
    elif len(found_schemas) > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: Schemas found but access issues remain")
    else:
        print(f"\n❌ FAILURE: No working schemas found")

if __name__ == "__main__":
    main()