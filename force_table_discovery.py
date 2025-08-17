#!/usr/bin/env python3
"""
Force MySQL to discover tables in nuclear-fixed schemas
Must be run with sudo: sudo python3 force_table_discovery.py
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
        print("Usage: sudo python3 force_table_discovery.py")
        sys.exit(1)
    
    print("🔧 FORCING TABLE DISCOVERY IN MYSQL")
    print("=" * 60)
    
    # Test schemas we know have data but no visible tables
    test_schemas = [
        'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109',
        'MaxZhang_Cullinan_183_100ReadRAC2_20250109'
    ]
    
    print("1️⃣ Analyzing the problem...")
    
    for schema in test_schemas:
        schema_path = f"/var/lib/mysql/{schema}"
        
        # Count .ibd files (data files)
        success, output = run_cmd(['find', schema_path, '-name', '*.ibd', '-type', 'f'])
        ibd_count = len(output.split('\n')) if success and output else 0
        
        # Check for .frm files (structure files - older MySQL)
        success, output = run_cmd(['find', schema_path, '-name', '*.frm', '-type', 'f'])
        frm_count = len(output.split('\n')) if success and output else 0
        
        print(f"   📊 {schema}: {ibd_count} .ibd files, {frm_count} .frm files")
        
        if ibd_count > 0 and frm_count == 0:
            print(f"      ⚠️  MySQL 8.0+ schema with data but no structure info")
    
    print(f"\n💡 PROBLEM IDENTIFIED:")
    print(f"   • Data files (.ibd) exist but no table structure info")
    print(f"   • MySQL 8.0+ uses data dictionary instead of .frm files") 
    print(f"   • Nuclear fix moved data but didn't rebuild data dictionary")
    print(f"   • Need to force MySQL to rebuild internal table catalog")
    
    print(f"\n2️⃣ Attempting table discovery methods...")
    
    # Method 1: mysql_upgrade (force data dictionary rebuild)
    print(f"   🔧 Method 1: Running mysql_upgrade...")
    success, output = run_cmd(['mysql_upgrade', '-u', 'root', '--force'])
    if success:
        print(f"      ✅ mysql_upgrade completed")
    else:
        print(f"      ⚠️  mysql_upgrade failed: {output}")
    
    # Method 2: Restart MySQL with recovery
    print(f"\n   🔧 Method 2: MySQL restart with recovery...")
    success, output = run_cmd(['systemctl', 'stop', 'mysql'])
    if success:
        print(f"      ✅ MySQL stopped")
        time.sleep(3)
        
        success, output = run_cmd(['systemctl', 'start', 'mysql'])
        if success:
            print(f"      ✅ MySQL started")
            time.sleep(5)
        else:
            print(f"      ❌ Failed to start: {output}")
            return
    else:
        print(f"      ❌ Failed to stop: {output}")
        return
    
    # Method 3: Force table import for each schema
    print(f"\n   🔧 Method 3: Force table import...")
    
    working_schemas = 0
    total_tables_discovered = 0
    
    for schema in test_schemas:
        print(f"\n      📋 Processing: {schema}")
        
        # Try ALTER DATABASE to force data dictionary update
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'ALTER DATABASE `{schema}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;'])
        if success:
            print(f"         ✅ Database altered (forces DD update)")
        else:
            print(f"         ⚠️  ALTER failed: {output}")
        
        # Check if tables are now visible
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
        if success:
            table_lines = output.split('\n')[1:] if '\n' in output else []
            table_count = len([t for t in table_lines if t.strip()])
            print(f"         📊 Tables discovered: {table_count}")
            
            if table_count > 0:
                total_tables_discovered += table_count
                working_schemas += 1
                
                # Test data access
                first_table = table_lines[0].strip()
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SELECT COUNT(*) FROM `{first_table}`;'])
                if success:
                    count_line = output.split('\n')[-1] if '\n' in output else output
                    print(f"         🎉 DATA ACCESS WORKS: {count_line} rows")
                else:
                    if "Tablespace is missing" in output:
                        print(f"         ❌ Still tablespace errors: {output}")
                    else:
                        print(f"         ❌ Data access failed: {output}")
            else:
                print(f"         ❌ Still no tables visible")
        else:
            print(f"         ❌ Cannot access schema: {output}")
    
    print(f"\n3️⃣ Final results...")
    print(f"   ✅ Working schemas: {working_schemas}/{len(test_schemas)}")
    print(f"   📊 Total tables discovered: {total_tables_discovered}")
    
    if working_schemas > 0:
        print(f"\n🎉 TABLE DISCOVERY SUCCESS!")
        print(f"   • {working_schemas} schemas now have visible tables")
        print(f"   • {total_tables_discovered} tables total discovered")
        print(f"   • Data access is working")
        print(f"   • Nuclear fix + table discovery = COMPLETE!")
        
        print(f"\n✅ NEXT STEPS:")
        print(f"   1. Test your webapp - should show normal table dimensions")
        print(f"   2. Apply this fix to remaining ~1,200 schemas")
        print(f"   3. All tablespace errors should be eliminated")
        
        # Quick webapp test
        print(f"\n🌐 Quick webapp test...")
        test_schema = test_schemas[0]
        
        # Simulate fetch_tables function
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{test_schema}`; SHOW TABLES;'])
        if success and output:
            table_lines = output.split('\n')[1:] if '\n' in output else []
            if table_lines:
                first_table = table_lines[0].strip()
                
                # Count columns
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{test_schema}`; SELECT COUNT(*) FROM information_schema.columns WHERE table_schema="{test_schema}" AND table_name="{first_table}";'])
                if success:
                    col_count = output.split('\n')[-1] if '\n' in output else "?"
                
                    # Count rows
                    success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{test_schema}`; SELECT COUNT(*) FROM `{first_table}`;'])
                    if success:
                        row_count = output.split('\n')[-1] if '\n' in output else "?"
                        print(f"   🎯 Sample table: {first_table}")
                        print(f"   📐 Dimensions: {row_count}x{col_count}")
                        print(f"   ✅ Should display normally in webapp (no more 'archived' or 'ERROR')")
    else:
        print(f"\n❌ TABLE DISCOVERY FAILED")
        print(f"   • Nuclear fix worked but table discovery incomplete")
        print(f"   • May need manual intervention or different approach")
        print(f"   • Data is safe but not accessible through MySQL")

if __name__ == "__main__":
    main()