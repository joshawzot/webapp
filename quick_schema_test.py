#!/usr/bin/env python3
"""
Quick test using basic MySQL commands
"""

import subprocess
import sys

def run_mysql_cmd(cmd):
    """Run a MySQL command and return result."""
    try:
        full_cmd = ['mysql', '-u', 'root', '-e', cmd]
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    print("🧪 QUICK SCHEMA TEST")
    print("=" * 40)
    
    # Test 1: Basic MySQL
    print("1️⃣ Testing MySQL connection...")
    success, result = run_mysql_cmd("SELECT 1;")
    if success:
        print("   ✅ MySQL works")
    else:
        print(f"   ❌ MySQL failed: {result}")
        return
    
    # Test 2: List databases
    print("\n2️⃣ Checking for moved schemas...")
    success, result = run_mysql_cmd("SHOW DATABASES;")
    if success:
        databases = result.split('\n')[1:]  # Skip header
        test_schemas = [
            'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109',
            'MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109'
        ]
        
        found = 0
        for schema in test_schemas:
            if schema in databases:
                print(f"   ✅ Found: {schema}")
                found += 1
            else:
                print(f"   ❌ Missing: {schema}")
        
        print(f"   📊 Found {found}/{len(test_schemas)} schemas")
        
        if found == 0:
            print("\n❌ No moved schemas found - fix didn't work")
            return
    else:
        print(f"   ❌ SHOW DATABASES failed: {result}")
        return
    
    # Test 3: Test table access
    print("\n3️⃣ Testing table access...")
    test_schema = 'MaxZhang_Cullinan_183_100ReadRAC2Reset_20250109'
    
    # Test SHOW TABLES
    success, result = run_mysql_cmd(f"USE `{test_schema}`; SHOW TABLES;")
    if success:
        tables = result.split('\n')[1:] if '\n' in result else []
        table_count = len([t for t in tables if t.strip()])
        print(f"   ✅ Found {table_count} tables in {test_schema}")
        
        if table_count > 0:
            # Test data access
            first_table = tables[0].strip()
            if first_table:
                success, result = run_mysql_cmd(f"USE `{test_schema}`; SELECT COUNT(*) FROM `{first_table}`;")
                if success:
                    count_line = result.split('\n')[-1] if '\n' in result else result
                    print(f"   ✅ DATA ACCESS WORKS: {count_line} rows in {first_table}")
                    print(f"\n🎉 SUCCESS! Tablespace errors are FIXED!")
                    print(f"   • Schema access restored")
                    print(f"   • Data accessible")
                    print(f"   • Ready to continue with remaining schemas")
                    
                    choice = input(f"\nContinue fixing remaining 1,477 schemas? (y/N): ").strip().lower()
                    if choice == 'y':
                        print(f"Run: sudo python3 simple_fix_schemas.py")
                        print(f"And answer 'y' when prompted to continue")
                    
                else:
                    if "Tablespace is missing" in result:
                        print(f"   ❌ STILL BROKEN: Tablespace error persists")
                        print(f"   Error: {result}")
                    else:
                        print(f"   ❌ Data access failed: {result}")
        else:
            print(f"   ⚠️  No tables found")
    else:
        if "Unknown database" in result:
            print(f"   ❌ Schema not found: {result}")
        else:
            print(f"   ❌ SHOW TABLES failed: {result}")

if __name__ == "__main__":
    main()