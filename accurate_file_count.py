#!/usr/bin/env python3
"""
Accurate count of ALL directories in each location
Don't rely on MySQL visibility for counting
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
    print("🔍 ACCURATE FILE LOCATION COUNT")
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
    
    # Count ALL directories in each location (not just MySQL-visible)
    locations = [
        ("/var/lib/mysql", "Primary"),
        ("/local/mysql/data", "Archive"), 
        ("/local/mysql_corrupted_backup", "Backup")
    ]
    
    total_found = 0
    
    for path, name in locations:
        print(f"\n📁 {name} ({path}):")
        
        # Get ALL directories (not filtered by MySQL visibility)
        success, output = run_cmd(['sudo', 'ls', '-1', path], capture_output=True)
        if success:
            all_dirs = output.split('\n') if output else []
            
            # Count MaxZhang directories
            maxzhang_dirs = [d for d in all_dirs if 'MaxZhang' in d]
            print(f"   📊 Total MaxZhang directories: {len(maxzhang_dirs)}")
            total_found += len(maxzhang_dirs)
            
            # Check how many are from our migration list
            migrated_here = []
            for schema in originally_migrated:
                if schema in all_dirs:
                    migrated_here.append(schema)
            
            print(f"   ✅ Originally migrated schemas here: {len(migrated_here)}")
            
            # Sample check for .ibd files
            if len(migrated_here) > 0:
                sample_schemas = migrated_here[:3]
                total_ibd = 0
                for schema in sample_schemas:
                    schema_path = f"{path}/{schema}"
                    success2, output2 = run_cmd(['sudo', 'find', schema_path, '-name', '*.ibd', '2>/dev/null'], capture_output=True)
                    if success2:
                        ibd_files = output2.split('\n') if output2.strip() else []
                        ibd_count = len([f for f in ibd_files if f.strip()])
                        total_ibd += ibd_count
                        print(f"   📄 {schema}: {ibd_count} .ibd files")
                
                if total_ibd > 0:
                    avg_ibd = total_ibd // len(sample_schemas)
                    estimated_total_ibd = avg_ibd * len(migrated_here)
                    print(f"   💾 Estimated total .ibd files: {estimated_total_ibd}")
        else:
            print(f"   ❌ Cannot access: {output}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   📋 Originally migrated: {len(originally_migrated)}")
    print(f"   👁️  Currently visible to MySQL: {len(current_maxzhang)}")
    print(f"   📁 Total directories found: {total_found}")
    print(f"   🔍 Missing/Invisible: {total_found - len(current_maxzhang)}")
    
    # Specific check: How many originally migrated schemas are now in primary?
    success, output = run_cmd(['sudo', 'ls', '-1', '/var/lib/mysql'], capture_output=True)
    if success:
        primary_dirs = output.split('\n') if output else []
        migrated_in_primary = []
        for schema in originally_migrated:
            if schema in primary_dirs:
                migrated_in_primary.append(schema)
        
        print(f"\n🎯 KEY METRICS:")
        print(f"   📁 Originally migrated schemas now in primary: {len(migrated_in_primary)}")
        print(f"   👁️  Of those, visible to MySQL: {len(current_maxzhang)}")
        print(f"   👻 Invisible (have files but MySQL can't see): {len(migrated_in_primary) - len(current_maxzhang)}")
        
        if len(migrated_in_primary) > len(current_maxzhang):
            invisible_count = len(migrated_in_primary) - len(current_maxzhang)
            print(f"\n💡 DIAGNOSIS:")
            print(f"   ✅ Nuclear fix WAS successful - {len(migrated_in_primary)} schemas in primary")
            print(f"   ❌ MySQL data dictionary corruption: {invisible_count} schemas invisible")
            print(f"   🔧 Need to force MySQL to discover {invisible_count} schemas")
            
            print(f"\n🎯 NEXT ACTION:")
            print(f"   • Files are in correct location (/var/lib/mysql)")
            print(f"   • {invisible_count} schemas need database entries created")
            print(f"   • Run 'CREATE DATABASE' for invisible schemas")
        else:
            print(f"\n❌ Nuclear fix failed - files not moved to primary")

if __name__ == "__main__":
    main()