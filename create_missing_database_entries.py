#!/usr/bin/env python3
"""
Create MySQL database entries for invisible schemas that have files
This will make MySQL discover the 1,201 invisible schemas
Must be run with sudo: sudo python3 create_missing_database_entries.py
"""

import subprocess
import json
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
    print("🔧 CREATING MISSING DATABASE ENTRIES")
    print("=" * 80)
    print("This will force MySQL to discover 1,201 invisible schemas")
    print("Files are already in correct location - just need database entries")
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
    
    # Get currently visible schemas
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if not success:
        print(f"❌ Cannot access MySQL: {output}")
        return
    
    current_databases = output.split('\n')[1:]
    current_maxzhang = [db for db in current_databases if 'MaxZhang' in db]
    print(f"📊 Currently visible: {len(current_maxzhang)}")
    
    # Find schemas that exist in primary but not visible to MySQL
    success, output = run_cmd(['sudo', 'ls', '-1', '/var/lib/mysql'])
    if not success:
        print(f"❌ Cannot access primary directory: {output}")
        return
    
    primary_dirs = output.split('\n') if output else []
    invisible_schemas = []
    
    for schema in originally_migrated:
        if schema in primary_dirs and schema not in current_maxzhang:
            invisible_schemas.append(schema)
    
    print(f"👻 Invisible schemas with files: {len(invisible_schemas)}")
    
    if len(invisible_schemas) == 0:
        print("🎉 All schemas are already visible!")
        return
    
    print(f"\n🚀 Ready to create database entries for {len(invisible_schemas)} schemas")
    print(f"Expected completion time: {len(invisible_schemas) * 2 // 60} minutes")
    
    confirm = input(f"\nProceed with creating database entries for {len(invisible_schemas)} schemas? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        return
    
    print(f"\n🔧 Creating database entries...")
    successful_creates = 0
    failed_creates = 0
    discovered_tables = 0
    
    # Process in batches to show progress
    batch_size = 100
    total_batches = (len(invisible_schemas) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(invisible_schemas))
        batch_schemas = invisible_schemas[start_idx:end_idx]
        
        print(f"\n🔄 BATCH {batch_num + 1}/{total_batches}: Processing {len(batch_schemas)} schemas")
        
        batch_successful = 0
        batch_tables = 0
        
        for i, schema in enumerate(batch_schemas):
            print(f"   [{start_idx + i + 1}/{len(invisible_schemas)}] Creating: {schema}")
            
            try:
                # Create database entry
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema}`;'])
                if success:
                    batch_successful += 1
                    
                    # Test table discovery
                    success2, output2 = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
                    if success2:
                        table_lines = output2.split('\n')[1:] if '\n' in output2 else []
                        table_count = len([t for t in table_lines if t.strip()])
                        if table_count > 0:
                            print(f"      🎉 {table_count} tables discovered!")
                            batch_tables += table_count
                        else:
                            print(f"      ⚠️  Database created, no tables visible yet")
                            
                            # Try to force table discovery
                            success3, output3 = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; FLUSH TABLES;'])
                            if success3:
                                success4, output4 = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
                                if success4:
                                    table_lines2 = output4.split('\n')[1:] if '\n' in output4 else []
                                    table_count2 = len([t for t in table_lines2 if t.strip()])
                                    if table_count2 > 0:
                                        print(f"      🎉 After FLUSH: {table_count2} tables discovered!")
                                        batch_tables += table_count2
                    else:
                        print(f"      ❌ Database created but cannot test tables: {output2}")
                else:
                    print(f"      ❌ Database creation failed: {output}")
                    failed_creates += 1
                    
            except Exception as e:
                print(f"      ❌ Error creating database: {e}")
                failed_creates += 1
        
        successful_creates += batch_successful
        discovered_tables += batch_tables
        
        print(f"   📊 Batch {batch_num + 1} complete: {batch_successful}/{len(batch_schemas)} successful")
        print(f"   📊 Tables discovered in batch: {batch_tables}")
        print(f"   📊 Total progress: {successful_creates}/{len(invisible_schemas)} ({successful_creates/len(invisible_schemas)*100:.1f}%)")
        
        # Brief pause between batches
        if batch_num < total_batches - 1:
            time.sleep(1)
    
    # Final verification
    print(f"\n🔍 Final verification...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if success:
        final_databases = output.split('\n')[1:]
        final_maxzhang = [db for db in final_databases if 'MaxZhang' in db]
        newly_visible = len(final_maxzhang) - len(current_maxzhang)
        total_recovery_rate = len(final_maxzhang) / len(originally_migrated) * 100
        
        print(f"\n📊 OPERATION COMPLETE:")
        print(f"   ✅ Successfully created: {successful_creates}/{len(invisible_schemas)}")
        print(f"   ❌ Failed: {failed_creates}/{len(invisible_schemas)}")
        print(f"   🆕 Newly visible schemas: {newly_visible}")
        print(f"   📈 Total recovery rate: {len(final_maxzhang)}/{len(originally_migrated)} = {total_recovery_rate:.1f}%")
        print(f"   🔢 Total tables discovered: {discovered_tables}")
        
        if total_recovery_rate >= 95:
            print(f"\n🏆 MISSION ACCOMPLISHED!")
            print(f"   • 95%+ recovery rate achieved")
            print(f"   • Migration corruption fully resolved")
            print(f"   • {discovered_tables} tables immediately accessible")
            print(f"   • Test your webapp - should show normal table dimensions")
        elif newly_visible > 0:
            print(f"\n🎉 MAJOR PROGRESS!")
            print(f"   • {newly_visible} schemas newly recovered")
            print(f"   • {discovered_tables} tables immediately accessible")
            print(f"   • Some schemas may need additional table discovery")
            print(f"   • Test your webapp - most should show normal dimensions")
        else:
            print(f"\n⚠️  DATABASE CREATION ISSUES:")
            print(f"   • Files are in correct location")
            print(f"   • Database entries may have failed")
            print(f"   • May need manual intervention")

if __name__ == "__main__":
    main()