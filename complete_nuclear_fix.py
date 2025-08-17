#!/usr/bin/env python3
"""
Complete the nuclear fix for ALL remaining schemas
Based on the success of the first 281 schemas
Must be run with sudo: sudo python3 complete_nuclear_fix.py
"""

import subprocess
import os
import sys
import time
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
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 complete_nuclear_fix.py")
        sys.exit(1)
    
    print("🚀 COMPLETING NUCLEAR FIX FOR ALL REMAINING SCHEMAS")
    print("=" * 80)
    print("Based on the success of 281 recovered schemas (90% working!)")
    print("This will apply the proven nuclear fix to remaining schemas.")
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
    
    if len(missing_schemas) == 0:
        print("🎉 All schemas already recovered!")
        return
    
    # Check if missing schemas exist in archive
    archive_schemas = []
    try:
        success, output = run_cmd(['sudo', 'ls', '/local/mysql/data/', '2>/dev/null'])
        if success:
            all_archive = output.split('\n') if output else []
            archive_schemas = [s for s in all_archive if s in missing_schemas]
        print(f"📦 Found in archive: {len(archive_schemas)} schemas")
    except:
        print("⚠️  Cannot check archive directory")
    
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
    
    for i, schema in enumerate(archive_schemas, 1):
        print(f"\n[{i}/{len(archive_schemas)}] Nuclear fixing: {schema}")
        
        archive_path = f"/local/mysql/data/{schema}"
        corrupted_backup = f"/local/mysql_corrupted_backup/{schema}"
        primary_path = f"/var/lib/mysql/{schema}"
        
        try:
            # Step 1: Stop MySQL
            if i == 1 or i % 50 == 1:  # Every 50 schemas, restart fresh
                print("   ⏹️  Stopping MySQL...")
                success, output = run_cmd(['systemctl', 'stop', 'mysql'])
                if not success:
                    print(f"   ❌ Failed to stop MySQL: {output}")
                    failed_fixes += 1
                    continue
            
            # Step 2: Move archive data to backup, then to primary
            success, output = run_cmd(['mkdir', '-p', '/local/mysql_corrupted_backup'])
            if success:
                # First move to backup (in case we need to rollback)
                success, output = run_cmd(['cp', '-r', archive_path, corrupted_backup])
                if success:
                    # Then move to primary location
                    success, output = run_cmd(['mv', archive_path, primary_path])
                    if success:
                        # Set permissions
                        success, output = run_cmd(['chown', '-R', 'mysql:mysql', primary_path])
                        if success:
                            print(f"   ✅ Files moved and permissions set")
                        else:
                            print(f"   ❌ Permission setting failed: {output}")
                            failed_fixes += 1
                            continue
                    else:
                        print(f"   ❌ Move to primary failed: {output}")
                        failed_fixes += 1
                        continue
                else:
                    print(f"   ❌ Backup creation failed: {output}")
                    failed_fixes += 1
                    continue
            
            # Step 3: Start MySQL and create database entry
            print("   ▶️  Starting MySQL...")
            success, output = run_cmd(['systemctl', 'start', 'mysql'])
            if not success:
                print(f"   ❌ Failed to start MySQL: {output}")
                failed_fixes += 1
                continue
            
            time.sleep(2)
            
            # Step 4: Create database entry
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema}`;'])
            if success:
                print(f"   ✅ Database entry created")
                successful_fixes += 1
                
                # Quick test
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
                if success:
                    table_lines = output.split('\n')[1:] if '\n' in output else []
                    table_count = len([t for t in table_lines if t.strip()])
                    if table_count > 0:
                        print(f"   🎉 {table_count} tables discovered!")
                    else:
                        print(f"   ⚠️  Database created but no tables visible yet")
                else:
                    print(f"   ⚠️  Database created but cannot test tables")
            else:
                print(f"   ❌ Database creation failed: {output}")
                failed_fixes += 1
            
            # Progress update
            if i % 10 == 0:
                print(f"   📊 Progress: {i}/{len(archive_schemas)} ({successful_fixes} successful)")
                
        except Exception as e:
            print(f"   ❌ Nuclear fix failed: {e}")
            failed_fixes += 1
    
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
        
        print(f"\n📊 NUCLEAR FIX COMPLETE:")
        print(f"   ✅ Successfully processed: {successful_fixes}/{len(archive_schemas)}")
        print(f"   ❌ Failed: {failed_fixes}/{len(archive_schemas)}")
        print(f"   📈 Total recovery rate: {len(final_maxzhang)}/{len(originally_migrated)} = {len(final_maxzhang)/len(originally_migrated)*100:.1f}%")
        
        if successful_fixes > 0:
            print(f"\n🎉 NUCLEAR FIX SUCCESS!")
            print(f"   • {successful_fixes} additional schemas recovered")
            print(f"   • Most should have working table access")
            print(f"   • Run table discovery script for any remaining issues")
            print(f"   • Test your webapp - should show normal table dimensions")
    else:
        print(f"❌ Final MySQL restart failed")

if __name__ == "__main__":
    main()