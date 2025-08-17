#!/usr/bin/env python3
"""
SIMPLE FIX: Move schemas back from archive to primary and restart MySQL
This must be run with sudo: sudo python3 simple_fix_schemas.py
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path

def run_cmd(cmd):
    """Run command and print output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 simple_fix_schemas.py")
        sys.exit(1)
    
    print("🔧 SIMPLE SCHEMA FIX")
    print("=" * 50)
    
    # Get migrated schemas from log
    migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
    migrated_schemas = []
    
    if migration_log_path.exists():
        with open(migration_log_path, 'r') as f:
            migration_log = json.load(f)
        
        for entry in migration_log:
            if (entry.get('action') == 'VERIFY' and 
                entry.get('status') == 'success'):
                migrated_schemas.append(entry.get('schema'))
    
    print(f"📋 Found {len(migrated_schemas)} schemas to fix")
    
    # Test with first 5 schemas
    test_schemas = migrated_schemas[:5]
    print(f"🧪 Testing with first {len(test_schemas)} schemas...")
    
    successful_fixes = 0
    
    for i, schema_name in enumerate(test_schemas, 1):
        print(f"\n[{i}/{len(test_schemas)}] Fixing: {schema_name}")
        
        primary_path = f"/var/lib/mysql/{schema_name}"
        archive_path = f"/local/mysql/data/{schema_name}"
        
        # Check if archive exists
        if not Path(archive_path).exists():
            print(f"   ❌ Archive not found: {archive_path}")
            continue
        
        # Check current state
        if Path(primary_path).is_symlink():
            print(f"   🔗 Removing symbolic link...")
            if run_cmd(['rm', primary_path]):
                print(f"   ✅ Symbolic link removed")
            else:
                print(f"   ❌ Failed to remove symbolic link")
                continue
        elif Path(primary_path).exists():
            print(f"   🗂️  Removing existing directory...")
            if run_cmd(['rm', '-rf', primary_path]):
                print(f"   ✅ Directory removed")
            else:
                print(f"   ❌ Failed to remove directory")
                continue
        
        # Move from archive to primary
        print(f"   📦 Moving from archive to primary...")
        if run_cmd(['mv', archive_path, primary_path]):
            print(f"   ✅ Moved successfully")
            
            # Set permissions
            if run_cmd(['chown', '-R', 'mysql:mysql', primary_path]):
                print(f"   ✅ Permissions set")
                successful_fixes += 1
            else:
                print(f"   ❌ Failed to set permissions")
        else:
            print(f"   ❌ Failed to move files")
    
    print(f"\n📊 RESULTS: {successful_fixes}/{len(test_schemas)} schemas fixed")
    
    if successful_fixes > 0:
        print(f"\n🔄 Restarting MySQL...")
        if run_cmd(['systemctl', 'stop', 'mysql']):
            time.sleep(3)
            if run_cmd(['systemctl', 'start', 'mysql']):
                print(f"✅ MySQL restarted")
                time.sleep(5)
                
                print(f"\n🧪 Testing schema access...")
                # Test one schema
                test_schema = test_schemas[0]
                try:
                    # Simple MySQL test
                    result = subprocess.run([
                        'mysql', '-u', 'root', '-e', 
                        f'USE `{test_schema}`; SHOW TABLES;'
                    ], capture_output=True, text=True, check=True)
                    
                    lines = result.stdout.strip().split('\n')
                    table_count = len(lines) - 1 if len(lines) > 1 else 0
                    print(f"   ✅ {test_schema}: {table_count} tables accessible")
                    
                    # Test data access
                    if table_count > 0:
                        first_table = lines[1] if len(lines) > 1 else None
                        if first_table:
                            result = subprocess.run([
                                'mysql', '-u', 'root', '-e',
                                f'USE `{test_schema}`; SELECT COUNT(*) FROM `{first_table}`;'
                            ], capture_output=True, text=True, check=True)
                            print(f"   ✅ Data access working: {result.stdout.strip()}")
                            
                            print(f"\n🎉 SUCCESS! Schema access restored!")
                            print(f"Continue with remaining {len(migrated_schemas) - len(test_schemas)} schemas? (y/N): ", end="")
                            
                            choice = input().strip().lower()
                            if choice == 'y':
                                print(f"Processing remaining schemas...")
                                remaining_schemas = migrated_schemas[len(test_schemas):]
                                
                                for i, schema_name in enumerate(remaining_schemas, len(test_schemas) + 1):
                                    print(f"\n[{i}/{len(migrated_schemas)}] Fixing: {schema_name}")
                                    
                                    primary_path = f"/var/lib/mysql/{schema_name}"
                                    archive_path = f"/local/mysql/data/{schema_name}"
                                    
                                    if Path(archive_path).exists():
                                        if Path(primary_path).is_symlink():
                                            run_cmd(['rm', primary_path])
                                        elif Path(primary_path).exists():
                                            run_cmd(['rm', '-rf', primary_path])
                                        
                                        if run_cmd(['mv', archive_path, primary_path]):
                                            run_cmd(['chown', '-R', 'mysql:mysql', primary_path])
                                            successful_fixes += 1
                                            
                                            # Progress update every 100 schemas
                                            if i % 100 == 0:
                                                print(f"   📊 Progress: {i}/{len(migrated_schemas)}")
                                
                                print(f"\n🎉 FINAL RESULT: {successful_fixes}/{len(migrated_schemas)} schemas fixed!")
                            else:
                                print(f"❌ Stopped at user request")
                        
                except Exception as e:
                    print(f"   ❌ MySQL test failed: {e}")
            else:
                print(f"❌ Failed to start MySQL")
        else:
            print(f"❌ Failed to stop MySQL")
    else:
        print(f"❌ No schemas were fixed successfully")

if __name__ == "__main__":
    main()