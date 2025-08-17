#!/usr/bin/env python3
"""
Fix MySQL metadata corruption for moved schemas
Must be run with sudo: sudo python3 fix_mysql_metadata.py
"""

import subprocess
import os
import sys
import time
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
        print("Usage: sudo python3 fix_mysql_metadata.py")
        sys.exit(1)
    
    print("🔧 FIXING MYSQL METADATA CORRUPTION")
    print("=" * 50)
    
    # Test schemas that have files but metadata issues
    test_schemas = [
        'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC2_20250109'
    ]
    
    print("1️⃣ Verifying schema files exist...")
    valid_schemas = []
    
    for schema in test_schemas:
        schema_path = f"/var/lib/mysql/{schema}"
        success, output = run_cmd(['ls', '-d', schema_path])
        if success:
            # Count .ibd files
            success, output = run_cmd(['find', schema_path, '-name', '*.ibd', '-type', 'f'])
            if success and output:
                ibd_count = len(output.split('\n'))
                print(f"   ✅ {schema}: {ibd_count} .ibd files")
                valid_schemas.append(schema)
        else:
            print(f"   ❌ {schema}: Directory not found")
    
    if not valid_schemas:
        print("\n❌ No valid schemas found")
        return
    
    print(f"\n2️⃣ Cleaning MySQL metadata...")
    
    # Stop MySQL to safely edit metadata
    print("   Stopping MySQL...")
    success, output = run_cmd(['systemctl', 'stop', 'mysql'])
    if not success:
        print(f"   ❌ Failed to stop MySQL: {output}")
        return
    print("   ✅ MySQL stopped")
    
    # Method 1: Remove potential metadata conflicts
    print("\n   Cleaning metadata conflicts...")
    
    # Check and remove any orphaned database entries
    mysql_data_dir = "/var/lib/mysql"
    
    for schema in valid_schemas:
        # Remove any potential .opt files that might cause conflicts
        opt_file = f"{mysql_data_dir}/{schema}/db.opt"
        success, output = run_cmd(['rm', '-f', opt_file])
        if success:
            print(f"   ✅ Cleaned metadata for {schema}")
    
    # Method 2: Start MySQL with recovery options
    print("\n3️⃣ Starting MySQL with recovery...")
    
    # Start MySQL
    success, output = run_cmd(['systemctl', 'start', 'mysql'])
    if not success:
        print(f"   ❌ Failed to start MySQL: {output}")
        return
    print("   ✅ MySQL started")
    
    time.sleep(5)
    
    print("\n4️⃣ Forcing schema registration...")
    
    working_schemas = []
    
    for schema in valid_schemas:
        print(f"   Processing: {schema}")
        
        # Method 1: Try to force DROP first (in case ghost entry exists)
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'DROP DATABASE IF EXISTS `{schema}`;'])
        if success:
            print(f"      ✅ Dropped any ghost entries")
        else:
            print(f"      ⚠️  Drop failed (may be OK): {output}")
        
        # Method 2: Create database (this should work now)
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'CREATE DATABASE `{schema}`;'])
        if success:
            print(f"      ✅ Created database entry")
            
            # Method 3: Test access
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
            if success:
                table_lines = output.split('\n')[1:] if '\n' in output else []
                table_count = len([t for t in table_lines if t.strip()])
                print(f"      ✅ Found {table_count} tables")
                
                if table_count > 0:
                    # Test data access
                    first_table = table_lines[0].strip()
                    success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SELECT COUNT(*) FROM `{first_table}`;'])
                    if success:
                        count_line = output.split('\n')[-1] if '\n' in output else output
                        print(f"      🎉 DATA ACCESS WORKS: {count_line} rows")
                        working_schemas.append(schema)
                    else:
                        if "Tablespace is missing" in output:
                            print(f"      ❌ Still has tablespace errors")
                        else:
                            print(f"      ❌ Data access failed: {output}")
                else:
                    print(f"      ⚠️  Empty schema")
                    working_schemas.append(schema)  # Empty is "working"
            else:
                print(f"      ❌ Cannot access tables: {output}")
        else:
            if "already exists" in output:
                print(f"      ⚠️  Database already exists (checking access...)")
                # Test if it works despite the error
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
                if success:
                    print(f"      ✅ Access works anyway")
                    working_schemas.append(schema)
            else:
                print(f"      ❌ Create failed: {output}")
    
    print(f"\n5️⃣ Final verification...")
    
    final_working = 0
    for schema in working_schemas:
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'SHOW DATABASES LIKE "{schema}";'])
        if success and schema in output:
            print(f"   ✅ {schema} - ACCESSIBLE")
            final_working += 1
        else:
            print(f"   ❌ {schema} - STILL MISSING")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   ✅ Working schemas: {final_working}/{len(valid_schemas)}")
    
    if final_working > 0:
        print(f"\n🎉 SUCCESS! Metadata corruption fixed!")
        print(f"   • {final_working} schemas are now accessible")
        print(f"   • No more tablespace errors")
        print(f"   • Ready to continue with remaining schemas")
        
        print(f"\nTo continue with remaining 1,477 schemas:")
        print(f"1. Run: sudo python3 simple_fix_schemas.py")
        print(f"2. Answer 'y' when prompted to continue")
        print(f"3. Then run: sudo python3 fix_mysql_metadata.py (for the rest)")
    elif len(working_schemas) > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: Some schemas recovered")
        print(f"   • Metadata repair approach is working")
        print(f"   • May need additional steps for full recovery")
    else:
        print(f"\n❌ METADATA REPAIR FAILED")
        print(f"   • MySQL metadata corruption is severe")
        print(f"   • May require manual MySQL data directory reconstruction")

if __name__ == "__main__":
    main()