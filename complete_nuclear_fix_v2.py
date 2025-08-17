#!/usr/bin/env python3
"""
Fixed version: Complete the nuclear fix for ALL remaining schemas
Now properly accesses the archive directory
Must be run with sudo: sudo python3 complete_nuclear_fix_v2.py
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path

def run_cmd(cmd, capture_output=True, shell=False):
    """Run command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True, shell=shell)
        return True, result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 complete_nuclear_fix_v2.py")
        sys.exit(1)
    
    print("🚀 NUCLEAR FIX V2: Processing Archive Schemas")
    print("=" * 80)
    print("Fixed to properly access archive directory")
    print("This will move 1,424 schemas from archive to primary and nuclear fix them.")
    print()
    
    # Get all schemas that should exist from migration log
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    if not migration_log_path.exists():
        print("❌ Migration log not found")
        return
    
    with open(migration_log_path, 'r') as f:
        migration_log = json.load(f)
    
    originally_migrated = []
    for entry in migration_log:
        if (entry.get('action') == 'VERIFY' and 
            entry.get('status') == 'success'):
            originally_migrated.append(entry.get('schema'))
    
    print(f"📋 Originally migrated: {len(originally_migrated)} schemas")
    
    # Get currently visible schemas
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if not success:
        print(f"❌ Cannot access MySQL: {output}")
        return
    
    current_databases = output.split('\n')[1:]  # Skip header
    current_maxzhang = [db for db in current_databases if 'MaxZhang' in db]
    
    print(f"📊 Currently visible: {len(current_maxzhang)} schemas")
    
    # Find missing schemas
    missing_schemas = []
    for schema in originally_migrated:
        if schema not in current_maxzhang:
            missing_schemas.append(schema)
    
    print(f"🔍 Missing schemas: {len(missing_schemas)}")
    
    # Check archive directory properly with explicit sudo ls
    print(f"📦 Checking archive directory...")
    success, output = run_cmd(['sudo', 'ls', '/local/mysql/data/'])
    if not success:
        print(f"❌ Cannot access archive: {output}")
        return
    
    all_archive_dirs = output.split('\n') if output else []
    archive_schemas = []
    
    # Find which missing schemas are actually in archive
    for schema in missing_schemas:
        if schema in all_archive_dirs:
            archive_schemas.append(schema)
    
    print(f"✅ Found in archive: {len(archive_schemas)} schemas")
    
    if len(archive_schemas) == 0:
        print("❌ No schemas found in archive to recover")
        return
    
    print(f"\n🚀 Ready to nuclear fix {len(archive_schemas)} schemas")
    print(f"Expected completion time: {len(archive_schemas) * 3 // 60} minutes")
    
    confirm = input(f"\nProceed with nuclear fix for {len(archive_schemas)} schemas? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        return
    
    print(f"\n💥 Starting nuclear fix...")
    successful_fixes = 0
    failed_fixes = 0
    
    # Create backup directory
    run_cmd(['mkdir', '-p', '/local/mysql_corrupted_backup'])
    
    # Process in batches to avoid too many MySQL restarts
    batch_size = 100
    total_batches = (len(archive_schemas) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(archive_schemas))
        batch_schemas = archive_schemas[start_idx:end_idx]
        
        print(f"\n🔄 BATCH {batch_num + 1}/{total_batches}: Processing {len(batch_schemas)} schemas")
        
        # Stop MySQL for batch processing
        print("   ⏹️  Stopping MySQL...")
        success, output = run_cmd(['systemctl', 'stop', 'mysql'])
        if not success:
            print(f"   ❌ Failed to stop MySQL: {output}")
            continue
        
        time.sleep(3)
        
        # Process all schemas in this batch
        batch_successful = 0
        for i, schema in enumerate(batch_schemas):
            print(f"   [{start_idx + i + 1}/{len(archive_schemas)}] Nuclear fixing: {schema}")
            
            archive_path = f"/local/mysql/data/{schema}"
            backup_path = f"/local/mysql_corrupted_backup/{schema}"
            primary_path = f"/var/lib/mysql/{schema}"
            
            try:
                # Step 1: Create backup in case we need to rollback
                success, output = run_cmd(['cp', '-r', archive_path, backup_path])
                if not success:
                    print(f"      ❌ Backup failed: {output}")
                    failed_fixes += 1
                    continue
                
                # Step 2: Move from archive to primary
                success, output = run_cmd(['mv', archive_path, primary_path])
                if not success:
                    print(f"      ❌ Move failed: {output}")
                    failed_fixes += 1
                    continue
                
                # Step 3: Set correct permissions
                success, output = run_cmd(['chown', '-R', 'mysql:mysql', primary_path])
                if not success:
                    print(f"      ❌ Permission setting failed: {output}")
                    failed_fixes += 1
                    continue
                
                print(f"      ✅ Files moved successfully")
                batch_successful += 1
                
            except Exception as e:
                print(f"      ❌ Nuclear fix failed: {e}")
                failed_fixes += 1
        
        # Start MySQL after batch
        print(f"   ▶️  Starting MySQL...")
        success, output = run_cmd(['systemctl', 'start', 'mysql'])
        if not success:
            print(f"   ❌ Failed to start MySQL: {output}")
            continue
        
        time.sleep(5)
        
        # Create database entries for successful moves in this batch
        database_successful = 0
        for i, schema in enumerate(batch_schemas):
            if i < batch_successful:  # Only for successfully moved schemas
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema}`;'])
                if success:
                    database_successful += 1
                    
                    # Quick table count test
                    success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
                    if success:
                        table_lines = output.split('\n')[1:] if '\n' in output else []
                        table_count = len([t for t in table_lines if t.strip()])
                        if table_count > 0:
                            print(f"      🎉 {schema}: {table_count} tables discovered!")
                        else:
                            print(f"      ⚠️  {schema}: Database created, tables pending discovery")
                else:
                    print(f"      ❌ Database creation failed for {schema}: {output}")
        
        successful_fixes += database_successful
        
        print(f"   📊 Batch {batch_num + 1} complete: {database_successful}/{len(batch_schemas)} successful")
        print(f"   📊 Total progress: {successful_fixes}/{len(archive_schemas)} ({successful_fixes/len(archive_schemas)*100:.1f}%)")
    
    # Final restart and summary
    print(f"\n🔄 Final MySQL restart...")
    success, output = run_cmd(['systemctl', 'stop', 'mysql'])
    time.sleep(3)
    success, output = run_cmd(['systemctl', 'start', 'mysql'])
    time.sleep(5)
    
    if success:
        print(f"✅ MySQL restarted")
        
        # Final count
        success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
        if success:
            final_databases = output.split('\n')[1:]
            final_maxzhang = [db for db in final_databases if 'MaxZhang' in db]
            print(f"📊 Final MaxZhang schema count: {len(final_maxzhang)}")
            
            # Calculate recovery rates
            original_visible = len(current_maxzhang)
            new_visible = len(final_maxzhang)
            newly_recovered = new_visible - original_visible
            total_recovery_rate = new_visible / len(originally_migrated) * 100
            
            print(f"\n📊 NUCLEAR FIX V2 COMPLETE:")
            print(f"   ✅ Successfully processed: {successful_fixes}/{len(archive_schemas)}")
            print(f"   ❌ Failed: {failed_fixes}/{len(archive_schemas)}")
            print(f"   🆕 Newly visible schemas: {newly_recovered}")
            print(f"   📈 Total recovery rate: {new_visible}/{len(originally_migrated)} = {total_recovery_rate:.1f}%")
            
            if successful_fixes > 0:
                print(f"\n🎉 MAJOR SUCCESS!")
                print(f"   • {newly_recovered} additional schemas recovered")
                print(f"   • Total visible schemas: {new_visible}")
                print(f"   • Most should have working table access")
                print(f"   • Test your webapp - should show normal table dimensions")
                
                if newly_recovered < len(archive_schemas):
                    print(f"\n⚠️  Some schemas still need attention:")
                    print(f"   • Run table discovery for schemas with 0 tables")
                    print(f"   • Check failed schemas manually")
    else:
        print(f"❌ Final MySQL restart failed")

if __name__ == "__main__":
    main()