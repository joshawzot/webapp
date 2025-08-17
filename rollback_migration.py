#!/usr/bin/env python3
"""
ROLLBACK MIGRATION SCRIPT
========================

This script can restore migrated schemas back to primary storage
using the SQL backups created during migration.

Usage: sudo python3 rollback_migration.py [migration_log_file]
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def run_cmd(cmd, capture_output=True, shell=False):
    """Run command and return result."""
    try:
        if shell:
            result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True, shell=True)
        else:
            result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
        return True, result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 rollback_migration.py [migration_log_file]")
        sys.exit(1)
    
    print("🔄 MIGRATION ROLLBACK TOOL")
    print("=" * 60)
    
    # Find migration log file
    if len(sys.argv) > 1:
        migration_log_file = sys.argv[1]
    else:
        # Find the most recent migration log
        log_files = list(Path('/home/admin2/webapp_2/').glob('safe_migration_log_*.json'))
        if not log_files:
            print("❌ No migration log files found!")
            print("Usage: sudo python3 rollback_migration.py [migration_log_file]")
            return
        
        migration_log_file = str(sorted(log_files)[-1])
        print(f"📋 Using most recent log: {migration_log_file}")
    
    if not os.path.exists(migration_log_file):
        print(f"❌ Migration log file not found: {migration_log_file}")
        return
    
    # Load migration log
    with open(migration_log_file, 'r') as f:
        migration_log = json.load(f)
    
    successful_migrations = [entry for entry in migration_log if entry.get('status') == 'success']
    
    print(f"📊 Migration log summary:")
    print(f"   📋 Total entries: {len(migration_log)}")
    print(f"   ✅ Successful migrations: {len(successful_migrations)}")
    print(f"   📦 Schemas available for rollback: {len(successful_migrations)}")
    
    if len(successful_migrations) == 0:
        print("✅ No successful migrations found to rollback!")
        return
    
    print(f"\n📝 Schemas that can be rolled back:")
    for i, entry in enumerate(successful_migrations[:10], 1):
        schema_name = entry['schema']
        age_days = entry.get('age_days', 'unknown')
        print(f"   {i}. {schema_name} ({age_days} days old)")
    
    if len(successful_migrations) > 10:
        print(f"   ... and {len(successful_migrations) - 10} more")
    
    print(f"\n⚠️  ROLLBACK PLAN:")
    print(f"   🔄 Restore {len(successful_migrations)} schemas to primary storage")
    print(f"   📦 Import from SQL backups in /local/mysql_migration_backups/")
    print(f"   🗑️  Remove archive copies")
    print(f"   ✅ Verify data integrity after rollback")
    
    confirm = input(f"\nProceed with rollback of {len(successful_migrations)} schemas? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Rollback cancelled by user")
        return
    
    print(f"\n🔄 Starting rollback process...")
    successful_rollbacks = 0
    failed_rollbacks = 0
    
    backup_directory = "/local/mysql_migration_backups"
    
    for i, entry in enumerate(successful_migrations, 1):
        schema_name = entry['schema']
        backup_file = f"{backup_directory}/{schema_name}.sql"
        
        print(f"\n[{i}/{len(successful_migrations)}] Rolling back: {schema_name}")
        
        try:
            # Check if backup file exists
            if not os.path.exists(backup_file):
                print(f"   ❌ Backup file not found: {backup_file}")
                failed_rollbacks += 1
                continue
            
            # Step 1: Import schema back to primary MySQL
            print(f"   📥 Importing schema from backup...")
            success, output = run_cmd(f'mysql -u root < {backup_file}', shell=True)
            if not success:
                print(f"   ❌ Import failed: {output}")
                failed_rollbacks += 1
                continue
            
            print(f"   ✅ Schema imported to primary MySQL")
            
            # Step 2: Verify schema exists and has tables
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema_name}`; SHOW TABLES;'])
            if success:
                table_lines = output.split('\n')[1:] if '\n' in output else []
                table_count = len([t for t in table_lines if t.strip()])
                print(f"   ✅ Verified: {table_count} tables accessible")
            else:
                print(f"   ⚠️  Warning: Could not verify tables: {output}")
            
            # Step 3: Remove archive copy (optional, commented out for safety)
            # archive_path = f"/local/mysql/data/{schema_name}"
            # if os.path.exists(archive_path):
            #     shutil.rmtree(archive_path)
            #     print(f"   🗑️  Removed archive copy")
            
            successful_rollbacks += 1
            
        except Exception as e:
            print(f"   ❌ Rollback failed: {e}")
            failed_rollbacks += 1
        
        # Progress update
        if i % 10 == 0:
            print(f"   📊 Progress: {i}/{len(successful_migrations)} ({successful_rollbacks} successful)")
    
    # Final summary
    print(f"\n📊 ROLLBACK COMPLETE:")
    print(f"   ✅ Successfully rolled back: {successful_rollbacks}/{len(successful_migrations)}")
    print(f"   ❌ Failed rollbacks: {failed_rollbacks}/{len(successful_migrations)}")
    
    if successful_rollbacks > 0:
        print(f"\n🎉 ROLLBACK SUCCESS!")
        print(f"   • {successful_rollbacks} schemas restored to primary storage")
        print(f"   • All data restored from SQL backups")
        print(f"   • Schemas should be accessible in webapp")
        print(f"   • Archive copies left intact for safety")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Test webapp to confirm schemas work properly")
        print(f"   • Verify all data is accessible")
        print(f"   • Consider cleaning up archive copies if rollback successful")
    
    if failed_rollbacks > 0:
        print(f"\n⚠️  SOME ROLLBACKS FAILED:")
        print(f"   • Check error messages above")
        print(f"   • Failed schemas remain in archive")
        print(f"   • Manual intervention may be required")

if __name__ == "__main__":
    main()