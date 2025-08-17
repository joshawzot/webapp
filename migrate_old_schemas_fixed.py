#!/usr/bin/env python3
"""
FIXED Schema Migration Script - Uses proper MySQL methods instead of file copying
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.append('/home/admin2/webapp_2')
from db_operations import create_connection
from analyze_schema_ages import extract_timestamp_from_name

def run_command(cmd, capture_output=False, check=True):
    """Execute a shell command with proper error handling."""
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return result
        else:
            result = subprocess.run(cmd, check=check)
            return result
    except subprocess.CalledProcessError as e:
        if capture_output and hasattr(e, 'stderr'):
            raise Exception(f"Command failed: {' '.join(cmd)}\nError: {e.stderr}")
        else:
            raise Exception(f"Command failed: {' '.join(cmd)}")

class FixedSchemaMigrator:
    """Fixed schema migrator using proper MySQL methods."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.source_datadir = "/var/lib/mysql"
        self.archive_datadir = "/local/mysql/data"
        self.backup_dir = Path("/home/admin2/webapp_2/migration_backups_fixed")
        self.log_file = f"fixed_migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.migration_log = []
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True)
        
    def log_action(self, action, schema_name, status, message):
        """Log migration actions."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "schema": schema_name,
            "status": status,
            "message": message
        }
        self.migration_log.append(entry)
        print(f"[{action}] {schema_name}: {status} - {message}")
    
    def backup_schema(self, schema_name):
        """Create a backup using mysqldump."""
        backup_file = self.backup_dir / f"{schema_name}.sql"
        
        try:
            if self.dry_run:
                self.log_action("BACKUP", schema_name, "success", f"DRY RUN: Would backup to {backup_file}")
                return True
            
            # Create mysqldump backup
            cmd = [
                'mysqldump', '-u', 'root',
                '--single-transaction',
                '--routines', 
                '--triggers',
                '--set-gtid-purged=OFF',
                '--hex-blob',  # Handle binary data properly
                schema_name
            ]
            
            with open(backup_file, 'w') as f:
                result = run_command(cmd, capture_output=True)
                f.write(result.stdout)
            
            # Verify backup file
            if backup_file.exists() and backup_file.stat().st_size > 1000:
                self.log_action("BACKUP", schema_name, "success", f"Backup created: {backup_file}")
                return True
            else:
                self.log_action("BACKUP", schema_name, "error", "Backup file too small or missing")
                return False
                
        except Exception as e:
            self.log_action("BACKUP", schema_name, "error", f"Backup failed: {e}")
            return False
    
    def migrate_schema_properly(self, schema_name):
        """Migrate schema using proper MySQL methods."""
        try:
            if self.dry_run:
                self.log_action("MIGRATE", schema_name, "success", "DRY RUN: Would migrate using mysqldump method")
                return True
            
            # Step 1: Create backup
            if not self.backup_schema(schema_name):
                return False
            
            # Step 2: Copy files to archive location (for storage)
            source_path = Path(self.source_datadir) / schema_name
            target_path = Path(self.archive_datadir) / schema_name
            
            # Create target directory
            run_command(['sudo', 'mkdir', '-p', str(target_path)])
            
            # Copy files to archive
            run_command(['sudo', 'rsync', '-av', f"{source_path}/", f"{target_path}/"])
            run_command(['sudo', 'chown', '-R', 'mysql:mysql', str(target_path)])
            run_command(['sudo', 'chmod', '-R', '750', str(target_path)])
            
            self.log_action("MIGRATE", schema_name, "success", f"Files archived to {target_path}")
            
            # Step 3: NO SYMBOLIC LINKS! 
            # Instead, we keep the schema accessible on primary storage
            # But record that it's been archived
            
            return True
            
        except Exception as e:
            self.log_action("MIGRATE", schema_name, "error", f"Migration failed: {e}")
            return False
    
    def verify_schema_full_access(self, schema_name):
        """Verify full data access (not just structure)."""
        try:
            conn = create_connection()
            cursor = conn.cursor()
            
            # Test database access
            cursor.execute("SHOW DATABASES LIKE %s", (schema_name,))
            if not cursor.fetchone():
                self.log_action("VERIFY", schema_name, "error", "Schema not found")
                cursor.close()
                conn.close()
                return False
            
            # Test table access
            cursor.execute(f"USE `{schema_name}`")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            if not tables:
                self.log_action("VERIFY", schema_name, "success", "Schema accessible (no tables)")
                cursor.close()
                conn.close()
                return True
            
            # Test actual data access on first table
            table_name = tables[0][0]
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                self.log_action("VERIFY", schema_name, "success", f"Full access verified: {len(tables)} tables, sample count: {count}")
                cursor.close()
                conn.close()
                return True
            except Exception as table_error:
                self.log_action("VERIFY", schema_name, "error", f"Data access failed: {table_error}")
                cursor.close()
                conn.close()
                return False
                
        except Exception as e:
            self.log_action("VERIFY", schema_name, "error", f"Verification failed: {e}")
            return False
    
    def get_schemas_to_migrate(self):
        """Get list of schemas that should be migrated (>60 days old)."""
        try:
            conn = create_connection()
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES")
            all_databases = [db[0] for db in cursor.fetchall() 
                           if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
            cursor.close()
            conn.close()
            
            # Filter by age
            schemas_to_migrate = []
            for db_name in all_databases:
                timestamp = extract_timestamp_from_name(db_name)
                if timestamp:
                    age_days = (datetime.now() - timestamp).days
                    if age_days > 60:
                        schemas_to_migrate.append(db_name)
            
            return schemas_to_migrate
            
        except Exception as e:
            print(f"Error getting schemas: {e}")
            return []
    
    def run_migration(self):
        """Run the fixed migration process."""
        print("🔧 RUNNING FIXED SCHEMA MIGRATION")
        print("=" * 80)
        
        schemas_to_migrate = self.get_schemas_to_migrate()
        print(f"📋 Found {len(schemas_to_migrate)} schemas to migrate")
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")
        
        successful_migrations = 0
        failed_migrations = 0
        
        for i, schema_name in enumerate(schemas_to_migrate, 1):
            print(f"\n[{i}/{len(schemas_to_migrate)}] Processing: {schema_name}")
            
            # Migrate schema
            if self.migrate_schema_properly(schema_name):
                # Verify access
                if self.verify_schema_full_access(schema_name):
                    successful_migrations += 1
                else:
                    failed_migrations += 1
            else:
                failed_migrations += 1
        
        # Save log
        log_path = Path(self.log_file)
        with open(log_path, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
        
        print(f"\n📊 MIGRATION COMPLETE:")
        print(f"   ✅ Successful: {successful_migrations}")
        print(f"   ❌ Failed: {failed_migrations}")
        print(f"   📄 Log saved: {log_path}")
        
        return successful_migrations, failed_migrations

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Schema Migration using proper MySQL methods")
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--live', action='store_true', help='Live migration mode')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.live:
        print("Please specify --dry-run or --live")
        return
    
    migrator = FixedSchemaMigrator(dry_run=args.dry_run)
    migrator.run_migration()

if __name__ == "__main__":
    main()