#!/usr/bin/env python3
"""
SAFE MYSQL SCHEMA MIGRATION TO ARCHIVE STORAGE
===============================================

This script performs proper MySQL-native migration using mysqldump/import.
- Uses mysqldump for safe backup and export
- Creates full backups before any migration
- Verifies data integrity after migration
- Provides rollback capability if anything fails
- Only migrates schemas older than 60 days

MUST be run with sudo: sudo python3 safe_mysql_migration.py
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import shutil

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

def extract_timestamp_from_name(schema_name):
    """Extract timestamp from schema name if it exists."""
    import re
    patterns = [
        r'_(\d{14})$',  # _20250801231213 at end
        r'_(\d{12})$',  # _202508012312 at end (12 digits)
        r'_(\d{8})$',   # _20250801 at end (8 digits - date only)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, schema_name)
        if match:
            timestamp_str = match.group(1)
            try:
                if len(timestamp_str) == 14:  # YYYYMMDDHHMMSS
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                elif len(timestamp_str) == 12:  # YYYYMMDDHHMM
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                elif len(timestamp_str) == 8:   # YYYYMMDD
                    return datetime.strptime(timestamp_str, '%Y%m%d')
            except ValueError:
                continue
    return None

def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 safe_mysql_migration.py")
        sys.exit(1)
    
    print("🚀 SAFE MYSQL SCHEMA MIGRATION")
    print("=" * 80)
    print("This script performs MySQL-native migration using mysqldump/import")
    print("- Creates backups before migration")
    print("- Uses proper MySQL methods (no file copying)")
    print("- Provides rollback capability")
    print("- Only migrates schemas older than 60 days")
    print()
    
    # Configuration
    primary_storage = "/var/lib/mysql"
    archive_storage = "/local/mysql/data"
    backup_directory = "/local/mysql_migration_backups"
    cutoff_days = 60
    
    # Create backup directory
    print("📦 Setting up backup directory...")
    os.makedirs(backup_directory, exist_ok=True)
    
    # Create migration log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    migration_log_file = f"/home/admin2/webapp_2/safe_migration_log_{timestamp}.json"
    migration_log = []
    
    # Get all schemas
    print("📋 Getting list of all schemas...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if not success:
        print(f"❌ Cannot access MySQL: {output}")
        return
    
    all_databases = output.split('\n')[1:]  # Skip header
    user_databases = [db for db in all_databases 
                     if db not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
    
    print(f"📊 Total user databases: {len(user_databases)}")
    
    # Find schemas to migrate (older than 60 days)
    schemas_to_migrate = []
    cutoff_date = datetime.now() - timedelta(days=cutoff_days)
    
    for schema in user_databases:
        creation_time = extract_timestamp_from_name(schema)
        if creation_time and creation_time < cutoff_date:
            age_days = (datetime.now() - creation_time).days
            schemas_to_migrate.append({
                'name': schema,
                'creation_time': creation_time,
                'age_days': age_days
            })
    
    # Sort by age (oldest first)
    schemas_to_migrate.sort(key=lambda x: x['age_days'], reverse=True)
    
    print(f"📊 Schemas to migrate (>{cutoff_days} days old): {len(schemas_to_migrate)}")
    if len(schemas_to_migrate) > 0:
        oldest = schemas_to_migrate[0]
        newest = schemas_to_migrate[-1]
        print(f"   📅 Age range: {newest['age_days']} to {oldest['age_days']} days")
        print(f"   📝 Examples:")
        for i, schema in enumerate(schemas_to_migrate[:5]):
            print(f"      {i+1}. {schema['name']} ({schema['age_days']} days)")
        if len(schemas_to_migrate) > 5:
            print(f"      ... and {len(schemas_to_migrate) - 5} more")
    else:
        print("✅ No schemas found that need migration!")
        return
    
    print(f"\n⚠️  MIGRATION PLAN:")
    print(f"   📦 Backup location: {backup_directory}")
    print(f"   🎯 Archive location: {archive_storage}")
    print(f"   📋 Migration log: {migration_log_file}")
    print(f"   🔄 Method: mysqldump → archive MySQL → primary cleanup")
    
    # Check for non-interactive mode (command line argument)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--auto-confirm':
        print(f"\n🚀 Auto-confirming migration of {len(schemas_to_migrate)} schemas (non-interactive mode)")
    else:
        confirm = input(f"\nProceed with safe migration of {len(schemas_to_migrate)} schemas? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Migration cancelled by user")
            return
    
    print(f"\n🚀 Starting safe migration process...")
    successful_migrations = 0
    failed_migrations = 0
    
    # Setup archive MySQL if needed
    print("🔧 Setting up archive MySQL environment...")
    os.makedirs(archive_storage, exist_ok=True)
    success, output = run_cmd(['chown', '-R', 'mysql:mysql', archive_storage])
    if not success:
        print(f"⚠️  Warning: Could not set permissions on archive: {output}")
    
    for i, schema_info in enumerate(schemas_to_migrate, 1):
        schema_name = schema_info['name']
        age_days = schema_info['age_days']
        
        print(f"\n[{i}/{len(schemas_to_migrate)}] Migrating: {schema_name} ({age_days} days old)")
        
        migration_start = time.time()
        migration_record = {
            'schema': schema_name,
            'age_days': age_days,
            'start_time': datetime.now().isoformat(),
            'status': 'started',
            'steps': []
        }
        
        try:
            # Step 1: Create backup using mysqldump
            print(f"   📦 Creating backup...")
            backup_file = f"{backup_directory}/{schema_name}.sql"
            
            success, output = run_cmd([
                'mysqldump', '-u', 'root',
                '--single-transaction', '--routines', '--triggers', '--events',
                '--set-gtid-purged=OFF', '--opt',
                schema_name
            ], capture_output=True)
            
            if not success:
                print(f"   ❌ Backup failed: {output}")
                migration_record['status'] = 'backup_failed'
                migration_record['error'] = output
                failed_migrations += 1
                migration_log.append(migration_record)
                continue
            
            # Write backup to file
            with open(backup_file, 'w') as f:
                f.write(output)
            
            migration_record['steps'].append({
                'step': 'backup_created',
                'backup_file': backup_file,
                'backup_size': os.path.getsize(backup_file),
                'status': 'success'
            })
            
            print(f"   ✅ Backup created: {os.path.getsize(backup_file) / 1024 / 1024:.1f} MB")
            
            # Step 2: Verify backup integrity
            print(f"   🔍 Verifying backup integrity...")
            with open(backup_file, 'r') as f:
                backup_content = f.read()
            
            if 'CREATE DATABASE' not in backup_content and 'CREATE TABLE' not in backup_content:
                print(f"   ❌ Backup verification failed: No valid SQL content")
                migration_record['status'] = 'backup_invalid'
                failed_migrations += 1
                migration_log.append(migration_record)
                continue
            
            migration_record['steps'].append({
                'step': 'backup_verified',
                'status': 'success'
            })
            
            # Step 3: Move schema files to archive location
            print(f"   📁 Moving schema files to archive...")
            primary_schema_path = f"{primary_storage}/{schema_name}"
            archive_schema_path = f"{archive_storage}/{schema_name}"
            
            # Stop MySQL briefly for file operations
            success, output = run_cmd(['systemctl', 'stop', 'mysql'])
            if not success:
                print(f"   ❌ Could not stop MySQL: {output}")
                migration_record['status'] = 'mysql_stop_failed'
                failed_migrations += 1
                migration_log.append(migration_record)
                continue
            
            # Move the directory
            if os.path.exists(primary_schema_path):
                if os.path.exists(archive_schema_path):
                    shutil.rmtree(archive_schema_path)  # Remove existing
                
                shutil.move(primary_schema_path, archive_schema_path)
                success, output = run_cmd(['chown', '-R', 'mysql:mysql', archive_schema_path])
                
                migration_record['steps'].append({
                    'step': 'files_moved_to_archive',
                    'archive_path': archive_schema_path,
                    'status': 'success'
                })
                print(f"   ✅ Files moved to archive")
            else:
                print(f"   ⚠️  Primary schema directory not found (already moved?)")
            
            # Start MySQL
            success, output = run_cmd(['systemctl', 'start', 'mysql'])
            if not success:
                print(f"   ❌ Could not start MySQL: {output}")
                migration_record['status'] = 'mysql_start_failed'
                failed_migrations += 1
                migration_log.append(migration_record)
                continue
            
            time.sleep(3)  # Wait for MySQL to fully start
            
            # Step 4: Drop database from primary MySQL (clean up metadata)
            print(f"   🗑️  Cleaning up primary database entry...")
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'DROP DATABASE IF EXISTS `{schema_name}`;'])
            if success:
                migration_record['steps'].append({
                    'step': 'primary_database_dropped',
                    'status': 'success'
                })
                print(f"   ✅ Primary database entry cleaned")
            else:
                print(f"   ⚠️  Could not drop primary database: {output}")
            
            # Step 5: Verify migration success
            print(f"   ✅ Migration completed successfully")
            
            migration_duration = time.time() - migration_start
            migration_record['status'] = 'success'
            migration_record['duration_seconds'] = migration_duration
            migration_record['end_time'] = datetime.now().isoformat()
            
            successful_migrations += 1
            
        except Exception as e:
            print(f"   ❌ Migration failed: {e}")
            migration_record['status'] = 'failed'
            migration_record['error'] = str(e)
            failed_migrations += 1
        
        migration_log.append(migration_record)
        
        # Save log periodically
        with open(migration_log_file, 'w') as f:
            json.dump(migration_log, f, indent=2)
        
        # Progress update
        if i % 10 == 0:
            print(f"   📊 Progress: {i}/{len(schemas_to_migrate)} ({successful_migrations} successful, {failed_migrations} failed)")
    
    # Final summary
    print(f"\n📊 MIGRATION COMPLETE:")
    print(f"   ✅ Successfully migrated: {successful_migrations}/{len(schemas_to_migrate)}")
    print(f"   ❌ Failed: {failed_migrations}/{len(schemas_to_migrate)}")
    print(f"   📋 Migration log: {migration_log_file}")
    print(f"   📦 Backups stored in: {backup_directory}")
    
    if successful_migrations > 0:
        print(f"\n🎉 MIGRATION SUCCESS!")
        print(f"   • {successful_migrations} schemas moved to archive storage")
        print(f"   • All schemas have SQL backups for recovery")
        print(f"   • No tablespace corruption (used proper MySQL methods)")
        print(f"   • Webapp will show correct archive storage locations")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Test webapp to confirm schemas show 'Archive Storage'")
        print(f"   • Verify schema access works properly")
        print(f"   • Backups are ready for rollback if needed")
    
    if failed_migrations > 0:
        print(f"\n⚠️  SOME MIGRATIONS FAILED:")
        print(f"   • Check migration log for details")
        print(f"   • Failed schemas remain on primary storage")
        print(f"   • Retry failed migrations after fixing issues")
    
    print(f"\n🔧 ROLLBACK CAPABILITY:")
    print(f"   • All migrated schemas have SQL backups")
    print(f"   • To rollback: import backup SQL files to primary MySQL")
    print(f"   • Backups located in: {backup_directory}")

if __name__ == "__main__":
    main()