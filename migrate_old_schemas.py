#!/usr/bin/env python3
"""
Schema Migration Script
Migrates old schemas (>60 days) from /dev/nvme0n1p3 to /dev/sda1 while maintaining webapp accessibility.
"""

import mysql.connector
import subprocess
import sys
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json
import re

# Import database configuration and analysis functions
sys.path.append('/home/admin2/webapp_2')
from db_operations import DB_CONFIG, create_connection
from analyze_schema_ages import extract_timestamp_from_name, get_schema_sizes, format_size

def run_command(cmd, shell=False, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"🔄 Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
            return result.stdout.strip() if result.stdout else ""
        else:
            subprocess.run(cmd, shell=shell, check=check)
            return ""
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        raise

class SchemaMigrator:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.source_datadir = "/var/lib/mysql"
        self.target_datadir = "/local/mysql/data"
        self.migration_log = []
        self.cutoff_date = datetime.now() - timedelta(days=60)
        
    def log_action(self, action, schema_name, status, details=""):
        """Log migration actions."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'schema': schema_name,
            'status': status,
            'details': details
        }
        self.migration_log.append(log_entry)
        
        status_icon = "✅" if status == "success" else "❌" if status == "error" else "ℹ️"
        print(f"{status_icon} {action}: {schema_name} - {status} {details}")
    
    def validate_environment(self):
        """Validate that the migration environment is ready."""
        print("🔍 Validating migration environment...")
        
        issues = []
        
        # Check MySQL connection and get datadir from MySQL itself
        try:
            conn = create_connection()
            cursor = conn.cursor()
            
            # Get MySQL's datadir setting
            cursor.execute("SELECT @@datadir")
            mysql_datadir = cursor.fetchone()[0]
            print(f"📁 MySQL datadir: {mysql_datadir}")
            
            # Test basic database operations
            cursor.execute("SHOW DATABASES")
            db_count = len(cursor.fetchall())
            print(f"✅ MySQL connection successful - found {db_count} databases")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            issues.append(f"MySQL connection failed: {e}")
        
        # Check target directory (create if doesn't exist in non-dry-run mode)
        target_parent = Path(self.target_datadir).parent
        if not target_parent.exists():
            if self.dry_run:
                print(f"ℹ️  Target parent directory does not exist: {target_parent}")
                print(f"   This is expected for dry-run mode")
            else:
                try:
                    target_parent.mkdir(parents=True, exist_ok=True)
                    print(f"📁 Created target parent directory: {target_parent}")
                except Exception as e:
                    issues.append(f"Cannot create target parent directory: {e}")
        
        # Check available space on target device (if accessible)
        try:
            if target_parent.exists():
                usage = shutil.disk_usage(target_parent)
                free_gb = usage.free / (1024**3)
                print(f"💾 Available space on target: {free_gb:.2f} GB")
                if free_gb < 100:  # Need at least 100GB for our 83GB migration
                    issues.append(f"Insufficient space on target: {free_gb:.2f} GB available, need ~100GB")
            else:
                print(f"ℹ️  Cannot check target disk space (directory doesn't exist yet)")
        except Exception as e:
            print(f"⚠️  Could not check target disk space: {e}")
        
        # For dry-run, we can be more lenient with file system checks
        if self.dry_run:
            print("ℹ️  Running in dry-run mode - some file system checks are relaxed")
        
        if issues:
            print("❌ Environment validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        print("✅ Environment validation passed")
        return True
    
    def get_schemas_to_migrate(self):
        """Get list of schemas that need to be migrated."""
        print("📊 Analyzing schemas for migration...")
        
        schema_sizes = get_schema_sizes()
        schemas_to_migrate = []
        
        for schema_name, size in schema_sizes.items():
            creation_time = extract_timestamp_from_name(schema_name)
            
            if creation_time and creation_time < self.cutoff_date:
                schemas_to_migrate.append({
                    'name': schema_name,
                    'size': size,
                    'creation_time': creation_time,
                    'age_days': (datetime.now() - creation_time).days
                })
        
        # Sort by size (largest first) for better progress indication
        schemas_to_migrate.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"📈 Found {len(schemas_to_migrate)} schemas to migrate")
        total_size = sum(s['size'] for s in schemas_to_migrate)
        print(f"💾 Total data to migrate: {format_size(total_size)}")
        
        return schemas_to_migrate
    
    def backup_schema(self, schema_name):
        """Create a backup of the schema before migration."""
        backup_file = f"/tmp/{schema_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        try:
            cmd = [
                'mysqldump',
                '-u', 'root',
                '--single-transaction',
                '--routines',
                '--triggers',
                '--events',
                '--set-gtid-purged=OFF',
                schema_name
            ]
            
            if not self.dry_run:
                with open(backup_file, 'w') as f:
                    subprocess.run(cmd, stdout=f, check=True)
            
            self.log_action("BACKUP", schema_name, "success", f"-> {backup_file}")
            return backup_file
            
        except Exception as e:
            self.log_action("BACKUP", schema_name, "error", str(e))
            return None
    
    def migrate_schema_files(self, schema_name):
        """Migrate schema files from source to target directory."""
        source_path = Path(self.source_datadir) / schema_name
        target_path = Path(self.target_datadir) / schema_name
        
        try:
            if self.dry_run:
                # For dry run, we'll estimate size using MySQL queries instead of file access
                conn = create_connection(schema_name)
                cursor = conn.cursor()
                
                # Get table information to estimate size
                cursor.execute("""
                    SELECT SUM(data_length + index_length) as total_size
                    FROM information_schema.TABLES 
                    WHERE table_schema = %s
                """, (schema_name,))
                
                result = cursor.fetchone()
                estimated_size = result[0] if result and result[0] else 0
                
                cursor.close()
                conn.close()
                
                self.log_action("MIGRATE_FILES", schema_name, "success", 
                              f"DRY RUN: {format_size(estimated_size)} -> {target_path}")
                return True
            else:
                # Live migration - check if source exists via MySQL first
                conn = create_connection()
                cursor = conn.cursor()
                cursor.execute("SHOW DATABASES LIKE %s", (schema_name,))
                if not cursor.fetchone():
                    cursor.close()
                    conn.close()
                    self.log_action("MIGRATE_FILES", schema_name, "error", "Schema not found in MySQL")
                    return False
                cursor.close()
                conn.close()
                
                # Now try file operations with sudo
                try:
                    # Use mysqldump approach for safer migration
                    dump_file = f"/tmp/{schema_name}_migration.sql"
                    
                    # Export schema
                    run_command([
                        'mysqldump', '-u', 'root', 
                        '--single-transaction', '--routines', '--triggers', 
                        '--set-gtid-purged=OFF', schema_name
                    ], capture_output=True)
                    
                    # Create target directory
                    run_command(['sudo', 'mkdir', '-p', str(target_path)])
                    
                    # Copy using MySQL's file operations or rsync with sudo
                    run_command([
                        'sudo', 'rsync', '-av', 
                        f"{source_path}/", f"{target_path}/"
                    ])
                    
                    # Set proper ownership and permissions
                    run_command(['sudo', 'chown', '-R', 'mysql:mysql', str(target_path)])
                    run_command(['sudo', 'chmod', '-R', '750', str(target_path)])
                    
                    # Get size estimate from MySQL
                    conn = create_connection(schema_name)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT SUM(data_length + index_length) as total_size
                        FROM information_schema.TABLES 
                        WHERE table_schema = %s
                    """, (schema_name,))
                    result = cursor.fetchone()
                    size = result[0] if result and result[0] else 0
                    cursor.close()
                    conn.close()
                    
                    self.log_action("MIGRATE_FILES", schema_name, "success", 
                                  f"{format_size(size)} -> {target_path}")
                    return True
                    
                except Exception as file_error:
                    self.log_action("MIGRATE_FILES", schema_name, "error", f"File operation failed: {file_error}")
                    return False
            
        except Exception as e:
            self.log_action("MIGRATE_FILES", schema_name, "error", str(e))
            return False
    
    def create_schema_link(self, schema_name):
        """Create a symbolic link from original location to new location."""
        source_path = Path(self.source_datadir) / schema_name
        target_path = Path(self.target_datadir) / schema_name
        
        try:
            if self.dry_run:
                self.log_action("CREATE_LINK", schema_name, "success", 
                              f"DRY RUN: {source_path} -> {target_path}")
                return True
            else:
                # Remove original directory with sudo
                run_command(['sudo', 'rm', '-rf', str(source_path)])
                
                # Create symbolic link with sudo
                run_command(['sudo', 'ln', '-s', str(target_path), str(source_path)])
                
                # Ensure MySQL can follow the link
                run_command(['sudo', 'chown', '-h', 'mysql:mysql', str(source_path)])
                
                self.log_action("CREATE_LINK", schema_name, "success", 
                              f"{source_path} -> {target_path}")
                return True
            
        except Exception as e:
            self.log_action("CREATE_LINK", schema_name, "error", str(e))
            return False
    
    def verify_schema_access(self, schema_name):
        """Verify that the migrated schema is still accessible via MySQL."""
        try:
            conn = create_connection()
            cursor = conn.cursor()
            
            # Test database access
            cursor.execute("SHOW DATABASES LIKE %s", (schema_name,))
            result = cursor.fetchone()
            
            if result:
                # Test table access
                cursor.execute(f"USE `{schema_name}`")
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
                self.log_action("VERIFY", schema_name, "success", 
                              f"{len(tables)} tables accessible")
                cursor.close()
                conn.close()
                return True
            else:
                self.log_action("VERIFY", schema_name, "error", "Schema not found")
                cursor.close()
                conn.close()
                return False
                
        except Exception as e:
            self.log_action("VERIFY", schema_name, "error", str(e))
            return False
    
    def migrate_single_schema(self, schema_info):
        """Migrate a single schema through the complete process."""
        schema_name = schema_info['name']
        
        print(f"\n🔄 Migrating: {schema_name} ({format_size(schema_info['size'])}, {schema_info['age_days']} days old)")
        
        # Step 1: Backup
        backup_file = self.backup_schema(schema_name)
        if not backup_file:
            return False
        
        # Step 2: Migrate files
        if not self.migrate_schema_files(schema_name):
            return False
        
        # Step 3: Create symbolic link
        if not self.create_schema_link(schema_name):
            return False
        
        # Step 4: Verify access
        if not self.verify_schema_access(schema_name):
            return False
        
        print(f"✅ Successfully migrated: {schema_name}")
        return True
    
    def run_migration(self, batch_size=10):
        """Run the complete migration process."""
        print("=" * 80)
        print(f"Schema Migration {'(DRY RUN)' if self.dry_run else '(LIVE)'}")
        print("=" * 80)
        
        # Validate environment
        if not self.validate_environment():
            return False
        
        # Get schemas to migrate
        schemas_to_migrate = self.get_schemas_to_migrate()
        if not schemas_to_migrate:
            print("ℹ️  No schemas found for migration")
            return True
        
        # Confirm migration
        if not self.dry_run:
            total_size = sum(s['size'] for s in schemas_to_migrate)
            print(f"\n⚠️  LIVE MIGRATION CONFIRMATION")
            print(f"📊 Schemas to migrate: {len(schemas_to_migrate)}")
            print(f"💾 Total data size: {format_size(total_size)}")
            print(f"🕒 Estimated time: {len(schemas_to_migrate) * 2:.0f} minutes")
            
            response = input("\nProceed with live migration? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Migration canceled by user")
                return False
        
        # Migrate schemas in batches
        successful_migrations = 0
        failed_migrations = 0
        
        for i in range(0, len(schemas_to_migrate), batch_size):
            batch = schemas_to_migrate[i:i+batch_size]
            
            print(f"\n📦 Processing batch {i//batch_size + 1}/{(len(schemas_to_migrate)-1)//batch_size + 1}")
            print("-" * 40)
            
            for schema_info in batch:
                if self.migrate_single_schema(schema_info):
                    successful_migrations += 1
                else:
                    failed_migrations += 1
        
        # Summary
        print("\n" + "=" * 80)
        print("MIGRATION SUMMARY")
        print("=" * 80)
        print(f"✅ Successful migrations: {successful_migrations}")
        print(f"❌ Failed migrations: {failed_migrations}")
        print(f"📊 Total schemas processed: {len(schemas_to_migrate)}")
        
        # Save migration log
        log_file = f"migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
        print(f"📄 Migration log saved: {log_file}")
        
        return failed_migrations == 0

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate old MySQL schemas to /dev/sda1')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Perform a dry run without making changes (default)')
    parser.add_argument('--live', action='store_true',
                       help='Perform live migration (opposite of --dry-run)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of schemas to process in each batch (default: 10)')
    
    args = parser.parse_args()
    
    # --live overrides --dry-run
    dry_run = not args.live
    
    migrator = SchemaMigrator(dry_run=dry_run)
    
    try:
        success = migrator.run_migration(batch_size=args.batch_size)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Migration interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Migration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())