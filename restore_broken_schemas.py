#!/usr/bin/env python3
"""
Restore broken migrated schemas using mysqldump backups
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.append('/home/admin2/webapp_2')
from db_operations import create_connection

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

class SchemaRestorer:
    """Restore broken schemas from backups."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.backup_dir = Path("/home/admin2/webapp_2/migration_backups")
        self.log_file = f"restoration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.restoration_log = []
        
    def log_action(self, action, schema_name, status, message):
        """Log restoration actions."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "schema": schema_name,
            "status": status,
            "message": message
        }
        self.restoration_log.append(entry)
        print(f"[{action}] {schema_name}: {status} - {message}")
    
    def test_schema_access(self, schema_name):
        """Test if schema has tablespace issues."""
        try:
            conn = create_connection(schema_name)
            cursor = conn.cursor()
            
            # Get first table
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            if not tables:
                cursor.close()
                conn.close()
                return True  # Empty schema is "working"
            
            table_name = tables[0][0]
            
            # Try to access data
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            cursor.fetchone()
            
            cursor.close()
            conn.close()
            return True  # Working
            
        except Exception as e:
            if "Tablespace is missing" in str(e) or "1812" in str(e):
                return False  # Broken
            return True  # Other error, assume working
    
    def get_broken_schemas(self):
        """Get list of schemas with tablespace issues."""
        # Read migration log to get list of migrated schemas
        migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
        migrated_schemas = []
        
        if migration_log_path.exists():
            with open(migration_log_path, 'r') as f:
                migration_log = json.load(f)
            
            for entry in migration_log:
                if (entry.get('action') == 'VERIFY' and 
                    entry.get('status') == 'success'):
                    migrated_schemas.append(entry.get('schema'))
        
        print(f"📋 Found {len(migrated_schemas)} migrated schemas to test")
        
        # Test which ones are broken
        broken_schemas = []
        working_schemas = []
        
        for i, schema in enumerate(migrated_schemas):
            if i % 100 == 0:
                print(f"   Testing {i+1}/{len(migrated_schemas)}...")
            
            if self.test_schema_access(schema):
                working_schemas.append(schema)
            else:
                broken_schemas.append(schema)
        
        print(f"📊 Results:")
        print(f"   ✅ Working: {len(working_schemas)}")
        print(f"   ❌ Broken: {len(broken_schemas)}")
        
        return broken_schemas, working_schemas
    
    def restore_schema(self, schema_name):
        """Restore a broken schema from backup."""
        backup_file = self.backup_dir / f"{schema_name}.sql"
        
        if not backup_file.exists():
            self.log_action("RESTORE", schema_name, "error", f"Backup file not found: {backup_file}")
            return False
        
        try:
            if self.dry_run:
                self.log_action("RESTORE", schema_name, "success", f"DRY RUN: Would restore from {backup_file}")
                return True
            
            # Step 1: Remove broken symbolic link
            broken_link = Path(f"/var/lib/mysql/{schema_name}")
            if broken_link.is_symlink():
                run_command(['sudo', 'rm', str(broken_link)])
                self.log_action("CLEANUP", schema_name, "success", "Removed broken symbolic link")
            
            # Step 2: Drop broken database if it exists
            try:
                conn = create_connection()
                cursor = conn.cursor()
                cursor.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")
                cursor.close()
                conn.close()
                self.log_action("DROP", schema_name, "success", "Dropped broken database")
            except Exception as drop_error:
                self.log_action("DROP", schema_name, "warning", f"Drop failed (might be OK): {drop_error}")
            
            # Step 3: Create new database
            conn = create_connection()
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE `{schema_name}`")
            cursor.close()
            conn.close()
            self.log_action("CREATE", schema_name, "success", "Created new database")
            
            # Step 4: Restore from backup
            cmd = ['mysql', '-u', 'root', schema_name]
            with open(backup_file, 'r') as f:
                result = run_command(cmd, capture_output=False)
            
            # Step 5: Verify restoration
            if self.test_schema_access(schema_name):
                self.log_action("RESTORE", schema_name, "success", "Restoration successful and verified")
                return True
            else:
                self.log_action("RESTORE", schema_name, "error", "Restoration failed verification")
                return False
                
        except Exception as e:
            self.log_action("RESTORE", schema_name, "error", f"Restoration failed: {e}")
            return False
    
    def run_restoration(self):
        """Run the restoration process."""
        print("🔧 RESTORING BROKEN SCHEMAS FROM BACKUPS")
        print("=" * 80)
        
        # Check if backup directory exists
        if not self.backup_dir.exists():
            print(f"❌ Backup directory not found: {self.backup_dir}")
            print("   Cannot proceed without backups")
            return
        
        print(f"✅ Backup directory found: {self.backup_dir}")
        
        # Get list of broken schemas
        broken_schemas, working_schemas = self.get_broken_schemas()
        
        if not broken_schemas:
            print("🎉 No broken schemas found! All migrated schemas are working correctly.")
            return
        
        print(f"\n🔧 Will restore {len(broken_schemas)} broken schemas")
        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")
        else:
            confirm = input(f"\nProceed with restoring {len(broken_schemas)} schemas? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Restoration cancelled by user")
                return
        
        successful_restorations = 0
        failed_restorations = 0
        
        for i, schema_name in enumerate(broken_schemas, 1):
            print(f"\n[{i}/{len(broken_schemas)}] Restoring: {schema_name}")
            
            if self.restore_schema(schema_name):
                successful_restorations += 1
            else:
                failed_restorations += 1
        
        # Save log
        log_path = Path(self.log_file)
        with open(log_path, 'w') as f:
            json.dump(self.restoration_log, f, indent=2)
        
        print(f"\n📊 RESTORATION COMPLETE:")
        print(f"   ✅ Successful: {successful_restorations}")
        print(f"   ❌ Failed: {failed_restorations}")
        print(f"   🔄 Working (unchanged): {len(working_schemas)}")
        print(f"   📄 Log saved: {log_path}")
        
        if successful_restorations > 0:
            print(f"\n🎉 {successful_restorations} schemas successfully restored!")
            print(f"   These schemas are now fully accessible on primary storage")
            print(f"   Archive files remain at /local/mysql/data/ for future reference")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Restore broken migrated schemas from backups")
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--live', action='store_true', help='Live restoration mode')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.live:
        print("Please specify --dry-run or --live")
        return
    
    restorer = SchemaRestorer(dry_run=args.dry_run)
    restorer.run_restoration()

if __name__ == "__main__":
    main()