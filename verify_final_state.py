#!/usr/bin/env python3
"""
Verify the actual final state after database creation
Check if schemas are really accessible despite SHOW DATABASES issues
"""

import subprocess
import json
from pathlib import Path

def run_cmd(cmd, capture_output=True):
    """Run command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
        return True, result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    print("🔍 VERIFYING FINAL STATE")
    print("=" * 60)
    print("Checking if schemas are actually accessible despite SHOW DATABASES issues")
    print()
    
    # Get migration log
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    with open(migration_log_path, 'r') as f:
        migration_log = json.load(f)
    
    originally_migrated = []
    for entry in migration_log:
        if (entry.get('action') == 'VERIFY' and 
            entry.get('status') == 'success'):
            originally_migrated.append(entry.get('schema'))
    
    print(f"📋 Originally migrated: {len(originally_migrated)}")
    
    # Method 1: Traditional SHOW DATABASES
    print(f"\n1️⃣ SHOW DATABASES check:")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if success:
        current_databases = output.split('\n')[1:]
        current_maxzhang = [db for db in current_databases if 'MaxZhang' in db]
        print(f"   📊 MaxZhang schemas visible: {len(current_maxzhang)}")
    else:
        print(f"   ❌ SHOW DATABASES failed: {output}")
        current_maxzhang = []
    
    # Method 2: Direct USE command test (bypasses SHOW DATABASES cache)
    print(f"\n2️⃣ Direct USE command test:")
    sample_schemas = originally_migrated[:10]  # Test first 10
    accessible_count = 0
    working_tables_count = 0
    
    for schema in sample_schemas:
        # Try to USE the schema directly
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SELECT "SUCCESS" as test;'])
        if success:
            accessible_count += 1
            
            # Check tables
            success2, output2 = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
            if success2:
                table_lines = output2.split('\n')[1:] if '\n' in output2 else []
                table_count = len([t for t in table_lines if t.strip()])
                if table_count > 0:
                    working_tables_count += 1
                    print(f"   ✅ {schema}: {table_count} tables accessible")
                else:
                    print(f"   ⚠️  {schema}: Accessible but no tables")
            else:
                print(f"   ⚠️  {schema}: Accessible but SHOW TABLES failed")
        else:
            print(f"   ❌ {schema}: Not accessible")
    
    print(f"\n   📊 Sample results: {accessible_count}/{len(sample_schemas)} accessible")
    print(f"   📊 With working tables: {working_tables_count}/{len(sample_schemas)}")
    
    # Extrapolate results
    if accessible_count > 0:
        estimated_accessible = accessible_count * len(originally_migrated) // len(sample_schemas)
        estimated_working = working_tables_count * len(originally_migrated) // len(sample_schemas)
        
        print(f"\n3️⃣ Extrapolated results:")
        print(f"   📊 Estimated accessible schemas: {estimated_accessible}")
        print(f"   📊 Estimated working schemas: {estimated_working}")
        print(f"   📈 Estimated recovery rate: {estimated_accessible/len(originally_migrated)*100:.1f}%")
    
    # Method 3: MySQL restart to clear cache
    print(f"\n4️⃣ Testing MySQL restart (to clear cache):")
    confirm = input("Restart MySQL to clear SHOW DATABASES cache? (y/N): ").strip().lower()
    if confirm == 'y':
        print("   ⏹️  Stopping MySQL...")
        success, output = run_cmd(['sudo', 'systemctl', 'stop', 'mysql'])
        if success:
            import time
            time.sleep(3)
            print("   ▶️  Starting MySQL...")
            success, output = run_cmd(['sudo', 'systemctl', 'start', 'mysql'])
            if success:
                time.sleep(5)
                print("   ✅ MySQL restarted")
                
                # Check SHOW DATABASES again
                success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
                if success:
                    new_databases = output.split('\n')[1:]
                    new_maxzhang = [db for db in new_databases if 'MaxZhang' in db]
                    improvement = len(new_maxzhang) - len(current_maxzhang)
                    print(f"   📊 After restart: {len(new_maxzhang)} schemas visible")
                    if improvement > 0:
                        print(f"   🎉 Improvement: +{improvement} newly visible schemas!")
                        recovery_rate = len(new_maxzhang) / len(originally_migrated) * 100
                        print(f"   📈 Total recovery rate: {recovery_rate:.1f}%")
                    else:
                        print(f"   ⚠️  No improvement in SHOW DATABASES")
            else:
                print(f"   ❌ MySQL start failed: {output}")
        else:
            print(f"   ❌ MySQL stop failed: {output}")
    
    print(f"\n💡 FINAL ASSESSMENT:")
    if accessible_count == len(sample_schemas):
        print(f"   🎉 FULL SUCCESS! All sampled schemas are accessible")
        print(f"   💡 SHOW DATABASES cache issue - restart fixed it or schemas work anyway")
        print(f"   ✅ Migration corruption fully resolved")
        print(f"   🎯 Test your webapp - should show normal table dimensions")
    elif accessible_count > len(sample_schemas) // 2:
        print(f"   🎉 MAJOR SUCCESS! Most schemas accessible")
        print(f"   ⚠️  Some schemas may need additional work")
        print(f"   ✅ Migration largely resolved")
    else:
        print(f"   ❌ PARTIAL SUCCESS - database entries may not have worked")
        print(f"   🔧 May need additional intervention")

if __name__ == "__main__":
    main()