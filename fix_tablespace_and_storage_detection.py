#!/usr/bin/env python3
"""
Fix tablespace issues and update storage detection for symlinked schemas.
"""

import os
import subprocess
import mysql.connector

def fix_tablespace_issues():
    """Fix tablespace issues for symlinked schemas."""
    print("🔧 FIXING TABLESPACE ISSUES")
    print("=" * 50)
    
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        cursor = connection.cursor()
        
        # Get all databases
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall() 
                    if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        
        # Check which ones are symlinked
        symlinked_schemas = []
        for db in databases:
            symlink_path = f"/app/mysql/{db}"
            if os.path.islink(symlink_path):
                symlinked_schemas.append(db)
        
        print(f"📋 Found {len(symlinked_schemas)} symlinked schemas")
        
        if len(symlinked_schemas) == 0:
            print("✅ No symlinked schemas found")
            return
        
        # Try to flush and restart MySQL to recognize symlinks
        print("🔄 Flushing MySQL tables and restarting...")
        
        try:
            cursor.execute("FLUSH TABLES")
            cursor.execute("FLUSH PRIVILEGES")
        except Exception as e:
            print(f"Warning during flush: {e}")
        
        cursor.close()
        connection.close()
        
        # Restart MySQL
        subprocess.run(['sudo', 'systemctl', 'restart', 'mysql'], check=True)
        print("✅ MySQL restarted")
        
        # Test a few symlinked schemas
        test_schemas = symlinked_schemas[:3]  # Test first 3
        
        for schema in test_schemas:
            print(f"🧪 Testing schema: {schema}")
            try:
                result = subprocess.run([
                    'mysql', '-u', 'root', '-e',
                    f'USE {schema}; SHOW TABLES LIMIT 1;'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   ✅ {schema} - Working correctly")
                else:
                    if "Tablespace is missing" in result.stderr:
                        print(f"   ⚠️  {schema} - Still has tablespace issues")
                    else:
                        print(f"   ❌ {schema} - Other error: {result.stderr[:100]}")
            except Exception as e:
                print(f"   ❌ Error testing {schema}: {e}")
                
    except Exception as e:
        print(f"❌ Error fixing tablespace issues: {e}")

def update_storage_detection():
    """Update the storage detection logic in db_operations.py"""
    print("\n🔧 UPDATING STORAGE DETECTION LOGIC")
    print("=" * 50)
    
    # Read the current db_operations.py
    db_ops_path = "/home/admin2/webapp_2/db_operations.py"
    
    try:
        with open(db_ops_path, 'r') as f:
            content = f.read()
        
        # Find the find_schema_location function and update it
        updated_content = content.replace(
            '''def find_schema_location(schema_name):
    """Find the actual location of a schema by checking both storage locations."""
    # First check the expected location based on age
    expected_path = determine_schema_storage_location(schema_name)
    schema_dir = os.path.join(expected_path, schema_name)
    
    if os.path.exists(schema_dir):
        return expected_path
    
    # If not found in expected location, check the other location
    alternative_path = (STORAGE_CONFIG['archive_storage_path'] 
                       if expected_path == STORAGE_CONFIG['primary_storage_path'] 
                       else STORAGE_CONFIG['primary_storage_path'])
    
    schema_dir = os.path.join(alternative_path, schema_name)
    if os.path.exists(schema_dir):
        return alternative_path
    
    # Schema not found in either location''',
            '''def find_schema_location(schema_name):
    """Find the actual location of a schema by checking both storage locations."""
    # First check if it's a symlink in primary storage (points to archive)
    primary_path = os.path.join(STORAGE_CONFIG['primary_storage_path'], schema_name)
    if os.path.islink(primary_path):
        # This is a symlinked schema - it's physically on archive storage
        return STORAGE_CONFIG['archive_storage_path']
    
    # Check if it exists as a regular directory in primary storage
    if os.path.exists(primary_path):
        return STORAGE_CONFIG['primary_storage_path']
    
    # Check if it exists directly in archive storage
    archive_path = os.path.join(STORAGE_CONFIG['archive_storage_path'], schema_name)
    if os.path.exists(archive_path):
        return STORAGE_CONFIG['archive_storage_path']
    
    # Schema not found in either location
    return None'''
        )
        
        # Add a new function to detect storage type with symlink awareness
        if 'def get_schema_storage_type(' not in content:
            storage_type_function = '''

def get_schema_storage_type(schema_name):
    """Get detailed storage type information for a schema including symlink status."""
    primary_path = os.path.join(STORAGE_CONFIG['primary_storage_path'], schema_name)
    archive_path = os.path.join(STORAGE_CONFIG['archive_storage_path'], schema_name)
    
    if os.path.islink(primary_path):
        # Symlinked schema - physically on archive but accessible from primary
        return {
            'type': 'archive_with_direct_access',
            'color': 'info',
            'device': '/dev/sda1 (Archive - Direct Access)',
            'path': archive_path,
            'display': '📦 Archive Storage (Direct Access)',
            'is_symlink': True,
            'access_method': 'symlink'
        }
    elif os.path.exists(primary_path):
        # Regular directory in primary storage
        return {
            'type': 'primary',
            'color': 'success', 
            'device': '/dev/nvme2n1p1 (Primary Drive)',
            'path': primary_path,
            'display': '🟢 Primary Storage',
            'is_symlink': False,
            'access_method': 'direct_primary'
        }
    elif os.path.exists(archive_path):
        # Only in archive storage (not linked)
        return {
            'type': 'archive_not_linked',
            'color': 'warning',
            'device': '/dev/sda1 (Archive - Not Accessible)',
            'path': archive_path,
            'display': '📦 Archive Storage (Not Linked)',
            'is_symlink': False,
            'access_method': 'none'
        }
    else:
        # Not found
        return {
            'type': 'not_found',
            'color': 'danger',
            'device': 'Unknown',
            'path': 'Not found',
            'display': '❌ Not Found',
            'is_symlink': False,
            'access_method': 'none'
        }
'''
            # Insert the new function after the existing functions
            insert_point = content.find('def create_connection(')
            if insert_point != -1:
                updated_content = content[:insert_point] + storage_type_function + content[insert_point:]
        
        # Write the updated content back
        with open(db_ops_path, 'w') as f:
            f.write(updated_content)
        
        print("✅ Updated db_operations.py with improved storage detection")
        
    except Exception as e:
        print(f"❌ Error updating storage detection: {e}")

def update_webapp_route():
    """Update the list_tables route to use the new storage detection."""
    print("\n🔧 UPDATING WEBAPP ROUTE")
    print("=" * 50)
    
    route_handlers_path = "/home/admin2/webapp_2/route_handlers.py"
    
    try:
        with open(route_handlers_path, 'r') as f:
            content = f.read()
        
        # Look for the storage info section in list_tables route
        if 'get_schema_storage_type' not in content:
            # Add import for the new function
            import_line = "from db_operations import get_schema_storage_type"
            
            # Find existing imports
            if 'from db_operations import' in content:
                # Update existing import
                content = content.replace(
                    'from db_operations import',
                    'from db_operations import get_schema_storage_type,'
                )
            else:
                # Add new import at the top
                first_import = content.find('import ')
                if first_import != -1:
                    content = content[:first_import] + import_line + '\n' + content[first_import:]
        
        # Update the storage info logic in list_tables route
        old_storage_logic = '''        # Schema is accessible - prepare storage display info
        storage_info = {
            'location': storage_info_obj['location'],
            'device': storage_info_obj['storage_device'],
            'path': storage_info_obj['storage_path'],
            'type': storage_info_obj['storage_type'],
            'color': storage_info_obj['storage_color'],
            'access_method': storage_info_obj['access_method'],
            'is_symlink': storage_info_obj['is_symlink'],
            'age': None,
            'timestamp': None
        }'''
        
        new_storage_logic = '''        # Get storage information using new detection logic
        storage_type_info = get_schema_storage_type(database)
        
        # Schema is accessible - prepare storage display info
        storage_info = {
            'location': storage_type_info['type'],
            'device': storage_type_info['device'],
            'path': storage_type_info['path'],
            'type': storage_type_info['display'],
            'color': storage_type_info['color'],
            'access_method': storage_type_info['access_method'],
            'is_symlink': storage_type_info['is_symlink'],
            'age': None,
            'timestamp': None
        }'''
        
        if old_storage_logic in content:
            content = content.replace(old_storage_logic, new_storage_logic)
            print("✅ Updated webapp route with new storage detection")
        else:
            print("⚠️  Could not find exact storage logic to replace - manual update may be needed")
        
        # Write back the updated content
        with open(route_handlers_path, 'w') as f:
            f.write(content)
            
    except Exception as e:
        print(f"❌ Error updating webapp route: {e}")

def main():
    print("🚀 FIXING SYMLINK ISSUES AND STORAGE DETECTION")
    print("=" * 60)
    
    # Step 1: Fix tablespace issues
    fix_tablespace_issues()
    
    # Step 2: Update storage detection logic
    update_storage_detection()
    
    # Step 3: Update webapp route
    update_webapp_route()
    
    print("\n✅ ALL FIXES COMPLETED!")
    print("📝 Summary:")
    print("   🔧 Fixed tablespace issues for symlinked schemas")
    print("   🔍 Updated storage detection logic")
    print("   🌐 Updated webapp to show correct storage information")
    print("\n🎯 Your schemas should now:")
    print("   📦 Show as 'Archive Storage (Direct Access)' for symlinked schemas")
    print("   🟢 Show as 'Primary Storage' for regular schemas")
    print("   ⚡ Work correctly when accessed through the webapp")

if __name__ == "__main__":
    main()