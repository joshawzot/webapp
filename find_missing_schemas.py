#!/usr/bin/env python3
"""
Detective script to find where the missing 1,201 schemas actually are
Must be run with sudo: sudo python3 find_missing_schemas.py
"""

import subprocess
import os
import sys
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
        print("Usage: sudo python3 find_missing_schemas.py")
        sys.exit(1)
    
    print("🔍 DETECTIVE WORK: Finding Missing Schemas")
    print("=" * 60)
    
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
    
    # Now let's look everywhere for these missing schemas
    locations_to_check = [
        ("/var/lib/mysql", "Primary MySQL"),
        ("/local/mysql/data", "Archive"),
        ("/local/mysql_corrupted_backup", "Nuclear Backup"),
        ("/tmp", "Temporary"),
    ]
    
    found_locations = {}
    
    for location, name in locations_to_check:
        print(f"\n🔍 Checking {name}: {location}")
        
        success, output = run_cmd(['ls', '-la', location], capture_output=True)
        if success:
            all_dirs = output.split('\n')
            dirs_only = []
            for line in all_dirs:
                if line.startswith('d') and not line.endswith('.') and not line.endswith('..'):
                    # Extract directory name (last part after spaces)
                    parts = line.split()
                    if len(parts) >= 9:
                        dir_name = parts[-1]
                        dirs_only.append(dir_name)
            
            # Check how many missing schemas are here
            found_here = []
            for schema in missing_schemas[:10]:  # Check first 10 as sample
                if schema in dirs_only:
                    found_here.append(schema)
            
            if found_here:
                # Extrapolate
                total_found = len(found_here) * len(missing_schemas) // 10
                found_locations[location] = total_found
                print(f"   ✅ Found {len(found_here)}/10 sample schemas")
                print(f"   📊 Estimated total: ~{total_found} schemas")
            else:
                print(f"   ❌ No missing schemas found")
        else:
            print(f"   ❌ Cannot access: {output}")
    
    # More specific check in primary location
    print(f"\n🔍 DETAILED CHECK: Primary MySQL directory")
    success, output = run_cmd(['ls', '/var/lib/mysql/'], capture_output=True)
    if success:
        all_mysql_dirs = output.split('\n')
        maxzhang_dirs = [d for d in all_mysql_dirs if 'MaxZhang' in d]
        print(f"   📊 All MaxZhang directories in /var/lib/mysql: {len(maxzhang_dirs)}")
        
        # Check if these directories have MySQL metadata
        invisible_schemas = []
        for schema in missing_schemas[:5]:  # Sample 5
            schema_path = f"/var/lib/mysql/{schema}"
            success, output = run_cmd(['ls', '-la', schema_path], capture_output=True)
            if success:
                # Check for .ibd files
                success2, output2 = run_cmd(['find', schema_path, '-name', '*.ibd', '|', 'wc', '-l'], capture_output=True)
                if success2:
                    ibd_count = output2.strip()
                    invisible_schemas.append((schema, ibd_count))
                    print(f"   📁 {schema}: {ibd_count} .ibd files (INVISIBLE to MySQL)")
        
        if invisible_schemas:
            print(f"   💡 Found {len(invisible_schemas)} schemas with data but invisible to MySQL!")
            print(f"   📊 These need metadata repair, not file movement")
    
    # Check corruption backup
    print(f"\n🔍 CORRUPTION BACKUP CHECK")
    success, output = run_cmd(['ls', '/local/mysql_corrupted_backup/', '2>/dev/null'], capture_output=True)
    if success:
        backup_schemas = output.split('\n') if output else []
        backup_maxzhang = [s for s in backup_schemas if 'MaxZhang' in s]
        print(f"   📦 Schemas in corruption backup: {len(backup_maxzhang)}")
    else:
        print(f"   ❌ No corruption backup directory or empty")
    
    print(f"\n📊 SUMMARY:")
    print(f"   🎯 Missing schemas: {len(missing_schemas)}")
    for location, count in found_locations.items():
        print(f"   📁 {location}: ~{count} schemas")
    
    print(f"\n💡 LIKELY SCENARIO:")
    if '/var/lib/mysql' in found_locations and found_locations['/var/lib/mysql'] > 0:
        print(f"   ✅ Schemas are in primary location but INVISIBLE to MySQL")
        print(f"   🔧 Need metadata repair, not file movement")
        print(f"   💥 Run nuclear fix on INVISIBLE schemas in primary location")
    elif '/local/mysql/data' in found_locations and found_locations['/local/mysql/data'] > 0:
        print(f"   📦 Schemas still in archive location")
        print(f"   📁 Need to move from archive to primary")
    else:
        print(f"   ❌ Schemas may be lost or in unexpected location")
        print(f"   🔍 Need deeper investigation")

if __name__ == "__main__":
    main()