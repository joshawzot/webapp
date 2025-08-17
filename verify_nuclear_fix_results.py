#!/usr/bin/env python3
"""
Verify what actually happened with the nuclear fix
Check file locations and MySQL visibility
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
    print("🔍 VERIFYING NUCLEAR FIX RESULTS")
    print("=" * 60)
    
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
    
    # Get currently visible to MySQL
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    current_databases = output.split('\n')[1:] if success else []
    current_maxzhang = [db for db in current_databases if 'MaxZhang' in db]
    print(f"📊 Currently visible to MySQL: {len(current_maxzhang)}")
    
    missing_schemas = [s for s in originally_migrated if s not in current_maxzhang]
    print(f"🔍 Still missing from MySQL: {len(missing_schemas)}")
    
    # Check where files actually are now
    locations = {
        "Primary": "/var/lib/mysql",
        "Archive": "/local/mysql/data", 
        "Backup": "/local/mysql_corrupted_backup"
    }
    
    file_locations = {}
    for name, path in locations.items():
        success, output = run_cmd(['sudo', 'ls', path], capture_output=True)
        if success:
            dirs = output.split('\n') if output else []
            maxzhang_dirs = [d for d in dirs if 'MaxZhang' in d]
            file_locations[name] = len(maxzhang_dirs)
            print(f"📁 {name} ({path}): {len(maxzhang_dirs)} MaxZhang directories")
        else:
            file_locations[name] = 0
            print(f"❌ {name} ({path}): Cannot access")
    
    # Sample check: Are missing schemas actually in primary location?
    print(f"\n🔍 SAMPLE CHECK: Missing schemas in primary location")
    sample_missing = missing_schemas[:5]  # Check first 5
    invisible_in_primary = 0
    
    for schema in sample_missing:
        primary_path = f"/var/lib/mysql/{schema}"
        success, output = run_cmd(['sudo', 'ls', '-la', primary_path], capture_output=True)
        if success:
            # Check for .ibd files
            success2, output2 = run_cmd(['sudo', 'find', primary_path, '-name', '*.ibd'], capture_output=True)
            if success2:
                ibd_files = output2.split('\n') if output2.strip() else []
                ibd_count = len([f for f in ibd_files if f.strip()])
                print(f"   📁 {schema}: {ibd_count} .ibd files (INVISIBLE to MySQL)")
                invisible_in_primary += 1
            else:
                print(f"   📁 {schema}: Directory exists but no .ibd files")
        else:
            print(f"   ❌ {schema}: Not in primary location")
    
    if invisible_in_primary > 0:
        estimated_invisible = invisible_in_primary * len(missing_schemas) // len(sample_missing)
        print(f"   💡 Estimated {estimated_invisible} schemas have files but are invisible to MySQL")
    
    print(f"\n📊 SITUATION ANALYSIS:")
    print(f"   📋 Total schemas to recover: {len(originally_migrated)}")
    print(f"   ✅ Currently visible: {len(current_maxzhang)}")
    print(f"   📁 Files in primary: {file_locations.get('Primary', 0)}")
    print(f"   📁 Files in archive: {file_locations.get('Archive', 0)}")
    print(f"   💾 Files in backup: {file_locations.get('Backup', 0)}")
    
    total_files = file_locations.get('Primary', 0)
    visible_count = len(current_maxzhang)
    
    if total_files > visible_count:
        invisible_count = total_files - visible_count
        print(f"\n💡 DIAGNOSIS:")
        print(f"   ✅ Nuclear fix moved files successfully")
        print(f"   ❌ {invisible_count} schemas have files but are invisible to MySQL")
        print(f"   🔧 Need MySQL data dictionary repair")
        
        print(f"\n🎯 RECOMMENDED ACTION:")
        print(f"   • Files are in correct location")
        print(f"   • Need to force MySQL to discover them")
        print(f"   • Run table discovery/metadata repair script")
    else:
        print(f"\n❌ FILES NOT MOVED PROPERLY")
        print(f"   • Nuclear fix may have failed silently")
        print(f"   • Need to investigate file movement")

if __name__ == "__main__":
    main()