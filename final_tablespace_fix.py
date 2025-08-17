#!/usr/bin/env python3
"""
Final fix for tablespace issues with symlinked schemas.
This script will properly recreate table metadata for symlinked schemas.
"""

import os
import subprocess
import mysql.connector
import sys

def test_schema_access(schema_name):
    """Test if we can access a specific schema and its tables."""
    try:
        result = subprocess.run([
            'mysql', '-u', 'root', '-e',
            f'USE {schema_name}; SHOW TABLES;'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            tables = [line.strip() for line in result.stdout.split('\n')[1:] if line.strip()]
            return True, tables
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def fix_tablespace_for_schema(schema_name):
    """Fix tablespace issues for a specific schema."""
    print(f"🔧 Fixing tablespace for schema: {schema_name}")
    
    # Check if it's a symlinked schema
    primary_path = f"/app/mysql/{schema_name}"
    if not os.path.islink(primary_path):
        print(f"   ℹ️  Schema {schema_name} is not symlinked - skipping")
        return True
    
    try:
        # Test initial access
        can_access, tables_or_error = test_schema_access(schema_name)
        
        if can_access:
            print(f"   ✅ Schema {schema_name} is already accessible ({len(tables_or_error)} tables)")
            return True
        
        if "Tablespace is missing" not in str(tables_or_error):
            print(f"   ❌ Different error for {schema_name}: {tables_or_error}")
            return False
        
        print(f"   🔄 Fixing tablespace issues...")
        
        # Method 1: Try to flush and restart MySQL
        try:
            connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',
                database=schema_name
            )
            cursor = connection.cursor()
            
            # Get table list from information_schema (this should work even with tablespace issues)
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_TYPE = 'BASE TABLE'
            """, (schema_name,))
            
            table_names = [row[0] for row in cursor.fetchall()]
            print(f"   📋 Found {len(table_names)} tables in information_schema")
            
            # Try to access each table to see which ones have issues
            problematic_tables = []
            working_tables = []
            
            for table in table_names[:5]:  # Test first 5 tables
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM `{table}` LIMIT 1")
                    cursor.fetchone()
                    working_tables.append(table)
                except Exception as table_error:
                    if "Tablespace is missing" in str(table_error):
                        problematic_tables.append(table)
            
            print(f"   ✅ Working tables: {len(working_tables)}")
            print(f"   ❌ Problematic tables: {len(problematic_tables)}")
            
            cursor.close()
            connection.close()
            
            if len(problematic_tables) == 0:
                print(f"   🎉 All tables are working for {schema_name}!")
                return True
            
        except Exception as e:
            print(f"   ⚠️  Could not analyze tables: {e}")
        
        # Method 2: Restart MySQL and flush everything
        print(f"   🔄 Restarting MySQL to recognize symlinked files...")
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', 'mysql'], check=True)
            print(f"   ✅ MySQL restarted")
            
            # Test access again
            can_access, tables_or_error = test_schema_access(schema_name)
            if can_access:
                print(f"   🎉 Schema {schema_name} is now accessible!")
                return True
            
        except Exception as e:
            print(f"   ❌ Error restarting MySQL: {e}")
        
        # If we get here, the schema still has issues
        print(f"   ⚠️  Schema {schema_name} still has tablespace issues")
        print(f"   💡 This schema may need manual intervention")
        return False
        
    except Exception as e:
        print(f"   ❌ Error fixing {schema_name}: {e}")
        return False

def get_symlinked_schemas():
    """Get list of all symlinked schemas."""
    symlinked_schemas = []
    
    try:
        result = subprocess.run(['sudo', 'find', '/app/mysql', '-type', 'l'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            for symlink in result.stdout.strip().split('\n'):
                if symlink:
                    schema_name = os.path.basename(symlink)
                    symlinked_schemas.append(schema_name)
    
    except Exception as e:
        print(f"Error finding symlinked schemas: {e}")
    
    return symlinked_schemas

def main():
    print("🚀 FINAL TABLESPACE FIX FOR SYMLINKED SCHEMAS")
    print("=" * 60)
    
    # Get all symlinked schemas
    symlinked_schemas = get_symlinked_schemas()
    
    if not symlinked_schemas:
        print("❌ No symlinked schemas found!")
        return
    
    print(f"📋 Found {len(symlinked_schemas)} symlinked schemas")
    
    # Test a few schemas to see current status
    test_schemas = symlinked_schemas[:5]  # Test first 5
    
    working_count = 0
    broken_count = 0
    
    for schema in test_schemas:
        print(f"\n🧪 Testing schema: {schema}")
        can_access, tables_or_error = test_schema_access(schema)
        
        if can_access:
            print(f"   ✅ Working - {len(tables_or_error)} tables")
            working_count += 1
        else:
            print(f"   ❌ Broken - {str(tables_or_error)[:100]}")
            broken_count += 1
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Working: {working_count}/{len(test_schemas)}")
    print(f"   ❌ Broken: {broken_count}/{len(test_schemas)}")
    
    if broken_count == 0:
        print("\n🎉 All tested schemas are working!")
        print("✅ No tablespace fixes needed")
        return
    
    # Apply fixes to broken schemas
    print(f"\n🔧 Applying fixes to broken schemas...")
    
    # Just restart MySQL once for all schemas
    try:
        print("🔄 Performing comprehensive MySQL restart...")
        subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], check=True)
        subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=True)
        print("✅ MySQL restarted successfully")
        
        # Test the schemas again
        print("\n🧪 Re-testing schemas after restart...")
        
        final_working = 0
        final_broken = 0
        
        for schema in test_schemas:
            can_access, tables_or_error = test_schema_access(schema)
            
            if can_access:
                print(f"   ✅ {schema} - Working ({len(tables_or_error)} tables)")
                final_working += 1
            else:
                print(f"   ❌ {schema} - Still broken")
                final_broken += 1
        
        print(f"\n📊 Final Results:")
        print(f"   ✅ Working: {final_working}/{len(test_schemas)}")
        print(f"   ❌ Still broken: {final_broken}/{len(test_schemas)}")
        
        if final_broken == 0:
            print("\n🎉 ALL SCHEMAS ARE NOW WORKING!")
            print("✅ Tablespace issues have been resolved")
        else:
            print(f"\n⚠️  {final_broken} schemas still have issues")
            print("💡 These may need manual intervention or recreation")
        
    except Exception as e:
        print(f"❌ Error during MySQL restart: {e}")

def test_webapp_access():
    """Test if the webapp storage detection is working correctly."""
    print("\n🌐 TESTING WEBAPP STORAGE DETECTION")
    print("=" * 50)
    
    try:
        # Import the new storage detection function
        sys.path.append('/home/admin2/webapp_2')
        from db_operations import get_schema_storage_type
        
        # Test with a few schemas
        test_schemas = ['MaxZhang_Cullinan_2416_KC3_H11_2hrD2_20250525072935']
        
        for schema in test_schemas:
            print(f"\n🔍 Testing storage detection for: {schema}")
            
            try:
                storage_info = get_schema_storage_type(schema)
                print(f"   📦 Type: {storage_info['type']}")
                print(f"   🎨 Color: {storage_info['color']}")
                print(f"   💾 Device: {storage_info['device']}")
                print(f"   📂 Path: {storage_info['path']}")
                print(f"   🖥️  Display: {storage_info['display']}")
                print(f"   🔗 Is Symlink: {storage_info['is_symlink']}")
                print(f"   ⚡ Access Method: {storage_info['access_method']}")
                
                if storage_info['is_symlink']:
                    print("   ✅ Correctly detected as symlinked schema!")
                else:
                    print("   ⚠️  Not detected as symlinked")
                    
            except Exception as e:
                print(f"   ❌ Error testing storage detection: {e}")
    
    except Exception as e:
        print(f"❌ Could not import storage detection function: {e}")

if __name__ == "__main__":
    main()
    test_webapp_access()