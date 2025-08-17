#!/usr/bin/env python3
"""
Help MySQL discover the moved schemas
Must be run with sudo: sudo python3 discover_moved_schemas.py
"""

import subprocess
import os
import sys
from pathlib import Path

def run_cmd(cmd, capture_output=True):
    """Run command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
        return True, result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 discover_moved_schemas.py")
        sys.exit(1)
    
    print("🔍 HELPING MYSQL DISCOVER MOVED SCHEMAS")
    print("=" * 50)
    
    # Test schemas that should have been moved
    test_schemas = [
        'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC2_20250109'
    ]
    
    print("1️⃣ Checking if schema directories exist...")
    existing_schemas = []
    
    for schema in test_schemas:
        schema_path = f"/var/lib/mysql/{schema}"
        success, output = run_cmd(['ls', '-ld', schema_path])
        if success:
            print(f"   ✅ Found: {schema}")
            existing_schemas.append(schema)
        else:
            print(f"   ❌ Missing: {schema}")
    
    if not existing_schemas:
        print("\n❌ No schema directories found in /var/lib/mysql/")
        print("The file move may have failed. Check manually:")
        print("sudo ls -la /var/lib/mysql/ | grep MaxZhang")
        return
    
    print(f"\n📊 Found {len(existing_schemas)} schema directories")
    
    print("\n2️⃣ Checking schema contents...")
    valid_schemas = []
    
    for schema in existing_schemas:
        schema_path = f"/var/lib/mysql/{schema}"
        
        # Check for .ibd files (data files)
        success, output = run_cmd(['find', schema_path, '-name', '*.ibd', '-type', 'f'])
        if success and output:
            ibd_count = len(output.split('\n'))
            print(f"   ✅ {schema}: {ibd_count} .ibd files")
            valid_schemas.append(schema)
        else:
            print(f"   ❌ {schema}: No .ibd files found")
    
    if not valid_schemas:
        print("\n❌ No valid schema data found")
        return
    
    print(f"\n3️⃣ Helping MySQL discover schemas...")
    
    # Method 1: Stop MySQL, restart, let it discover
    print("   Stopping MySQL...")
    success, output = run_cmd(['systemctl', 'stop', 'mysql'])
    if success:
        print("   ✅ MySQL stopped")
    else:
        print(f"   ❌ Failed to stop MySQL: {output}")
        return
    
    # Wait a moment
    import time
    time.sleep(3)
    
    print("   Starting MySQL...")
    success, output = run_cmd(['systemctl', 'start', 'mysql'])
    if success:
        print("   ✅ MySQL started")
    else:
        print(f"   ❌ Failed to start MySQL: {output}")
        return
    
    # Wait for MySQL to fully start
    time.sleep(5)
    
    print("\n4️⃣ Testing schema discovery...")
    
    for schema in valid_schemas:
        print(f"   Testing: {schema}")
        
        # Try to access the schema
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'SHOW DATABASES LIKE "{schema}";'])
        if success and schema in output:
            print(f"   ✅ MySQL can see {schema}")
            
            # Test table access
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
            if success:
                table_lines = output.split('\n')[1:] if '\n' in output else []
                table_count = len([t for t in table_lines if t.strip()])
                print(f"      📊 Found {table_count} tables")
                
                if table_count > 0:
                    # Test data access
                    first_table = table_lines[0].strip() if table_lines else None
                    if first_table:
                        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SELECT COUNT(*) FROM `{first_table}`;'])
                        if success:
                            print(f"      ✅ Data access works!")
                        else:
                            if "Tablespace is missing" in output:
                                print(f"      ❌ Still has tablespace errors")
                            else:
                                print(f"      ❌ Data access failed: {output}")
            else:
                print(f"      ❌ Cannot show tables: {output}")
        else:
            print(f"   ❌ MySQL cannot see {schema}")
            
            # Try to manually create the database entry
            print(f"      Trying to register schema...")
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema}`;'])
            if success:
                print(f"      ✅ Schema registered")
                
                # Test again
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
                if success:
                    print(f"      ✅ Now accessible!")
                else:
                    print(f"      ❌ Still not accessible: {output}")
            else:
                print(f"      ❌ Failed to register: {output}")
    
    print(f"\n5️⃣ Final test...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if success:
        databases = output.split('\n')[1:]  # Skip header
        found_count = 0
        for schema in valid_schemas:
            if schema in databases:
                found_count += 1
                print(f"   ✅ {schema} - WORKING")
            else:
                print(f"   ❌ {schema} - MISSING")
        
        print(f"\n📊 RESULTS: {found_count}/{len(valid_schemas)} schemas working")
        
        if found_count > 0:
            print(f"\n🎉 SUCCESS! Schema discovery worked!")
            print(f"   • {found_count} schemas are now accessible")
            print(f"   • MySQL can see the moved data")
            print(f"   • Ready to continue with remaining schemas")
        else:
            print(f"\n❌ Discovery failed - schemas still not visible to MySQL")
    else:
        print(f"   ❌ Cannot test final result: {output}")

if __name__ == "__main__":
    main()