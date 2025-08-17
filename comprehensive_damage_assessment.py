#!/usr/bin/env python3
"""
Comprehensive assessment of the schema situation
"""

import subprocess
import json
from pathlib import Path

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    print("🔍 COMPREHENSIVE DAMAGE ASSESSMENT")
    print("=" * 80)
    
    print("1️⃣ Checking ALL schemas in MySQL...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if success:
        all_databases = output.split('\n')[1:]  # Skip header
        user_databases = [db for db in all_databases if db not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        
        print(f"   📊 Total user databases: {len(user_databases)}")
        
        # Categorize databases
        maxzhang_schemas = [db for db in user_databases if 'MaxZhang' in db]
        other_schemas = [db for db in user_databases if 'MaxZhang' not in db]
        
        print(f"   📊 MaxZhang schemas: {len(maxzhang_schemas)}")
        print(f"   📊 Other schemas: {len(other_schemas)}")
    else:
        print(f"   ❌ Cannot access MySQL: {output}")
        return
    
    print(f"\n2️⃣ Testing non-MaxZhang schemas (should be working)...")
    working_other = 0
    
    for schema in other_schemas[:5]:  # Test first 5
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
        if success:
            table_lines = output.split('\n')[1:] if '\n' in output else []
            table_count = len([t for t in table_lines if t.strip()])
            
            if table_count > 0:
                working_other += 1
                print(f"   ✅ {schema}: {table_count} tables")
                
                # Test data access
                first_table = table_lines[0].strip()
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SELECT COUNT(*) FROM `{first_table}`;'])
                if success:
                    print(f"      🎉 Data access works")
                else:
                    print(f"      ❌ Data access failed: {output}")
            else:
                print(f"   ⚠️  {schema}: No tables")
        else:
            print(f"   ❌ {schema}: Cannot access")
    
    print(f"\n3️⃣ Testing MaxZhang schemas (migrated/fixed)...")
    working_maxzhang = 0
    
    for schema in maxzhang_schemas[:10]:  # Test first 10
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
        if success:
            table_lines = output.split('\n')[1:] if '\n' in output else []
            table_count = len([t for t in table_lines if t.strip()])
            
            if table_count > 0:
                working_maxzhang += 1
                print(f"   ✅ {schema}: {table_count} tables")
                
                # Test data access
                first_table = table_lines[0].strip()
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SELECT COUNT(*) FROM `{first_table}`;'])
                if success:
                    print(f"      🎉 Data access works")
                else:
                    print(f"      ❌ Data access failed: {output}")
            else:
                print(f"   ❌ {schema}: No tables (data dictionary issue)")
        else:
            print(f"   ❌ {schema}: Cannot access")
    
    print(f"\n4️⃣ Checking migration log...")
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    if migration_log_path.exists():
        with open(migration_log_path, 'r') as f:
            migration_log = json.load(f)
        
        migrated_count = len([entry for entry in migration_log 
                            if entry.get('action') == 'VERIFY' and entry.get('status') == 'success'])
        print(f"   📋 Originally migrated: {migrated_count} schemas")
        print(f"   📊 Currently visible: {len(maxzhang_schemas)} schemas")
        print(f"   📈 Recovery rate: {len(maxzhang_schemas)}/{migrated_count} = {len(maxzhang_schemas)/migrated_count*100:.1f}%")
    
    print(f"\n5️⃣ Checking backup locations...")
    # Check if original archive data still exists
    try:
        success, output = run_cmd(['sudo', 'ls', '/local/mysql/data/', '2>/dev/null'])
        if success:
            archive_schemas = output.split('\n') if output else []
            remaining_archive = len([s for s in archive_schemas if 'MaxZhang' in s])
            print(f"   📦 Remaining in archive: {remaining_archive} schemas")
        else:
            print(f"   📦 Archive directory empty or inaccessible")
    except:
        print(f"   📦 Cannot check archive")
    
    try:
        success, output = run_cmd(['sudo', 'ls', '/local/mysql_corrupted_backup/', '2>/dev/null'])
        if success:
            backup_schemas = output.split('\n') if output else []
            backed_up = len(backup_schemas)
            print(f"   💾 Nuclear fix backups: {backed_up} schemas")
        else:
            print(f"   💾 No nuclear fix backups")
    except:
        print(f"   💾 Cannot check backups")
    
    print(f"\n📊 DAMAGE ASSESSMENT SUMMARY:")
    print(f"   ✅ Non-migrated schemas working: {working_other}/{min(len(other_schemas), 5)}")
    print(f"   ❌ Migrated schemas with tables: {working_maxzhang}/{min(len(maxzhang_schemas), 10)}")
    print(f"   📊 Schemas recovered: {len(maxzhang_schemas)} visible")
    print(f"   🔧 Schemas need table repair: {len(maxzhang_schemas) - working_maxzhang}")
    
    print(f"\n💡 SITUATION ANALYSIS:")
    if working_other > 0:
        print(f"   ✅ MySQL is fundamentally working (non-migrated schemas OK)")
    else:
        print(f"   ❌ MySQL has broader issues")
    
    if working_maxzhang > 0:
        print(f"   ✅ Some migrated schemas completely recovered")
        print(f"   💡 SOLUTION: Apply same fix to remaining schemas")
    else:
        print(f"   ❌ NO migrated schemas have working tables")
        print(f"   💡 PROBLEM: Data dictionary corruption is complete")
    
    print(f"\n🎯 RECOMMENDED NEXT ACTIONS:")
    if working_maxzhang > 0:
        print(f"   1. ✅ Success pattern identified - scale it up")
        print(f"   2. 🔧 Apply working fix to remaining {len(maxzhang_schemas) - working_maxzhang} schemas")
    elif working_other > 0:
        print(f"   1. 🔧 MySQL working but migration recovery failed")
        print(f"   2. 💾 Consider restoring from pre-migration state")
        print(f"   3. 📋 Use different migration approach")
    else:
        print(f"   1. 🚨 Critical MySQL issues - broader than migration")
        print(f"   2. 🛠️ Full MySQL repair needed")
    
    if working_maxzhang == 0 and len(maxzhang_schemas) > 0:
        print(f"\n🚨 CRITICAL DATA DICTIONARY CORRUPTION:")
        print(f"   • {len(maxzhang_schemas)} schemas exist but have no tables")
        print(f"   • Physical data (.ibd files) present but metadata lost")
        print(f"   • This requires advanced recovery techniques")

if __name__ == "__main__":
    main()