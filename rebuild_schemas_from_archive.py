#!/usr/bin/env python3
"""
Rebuild schemas completely by dropping and recreating them from archive data
This is the only way to fix severely corrupted InnoDB metadata
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

class SchemaRebuilder:
    """Rebuild schemas completely from archive data."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.primary_datadir = "/var/lib/mysql"
        self.archive_datadir = "/local/mysql/data"
        self.log_file = f"rebuild_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.rebuild_log = []
        
    def log_action(self, action, schema_name, status, message):
        """Log rebuild actions."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "schema": schema_name,
            "status": status,
            "message": message
        }
        self.rebuild_log.append(entry)
        print(f"[{action}] {schema_name}: {status} - {message}")
    
    def get_migrated_schemas(self):
        """Get list of all migrated schemas."""
        migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
        migrated_schemas = []
        
        if migration_log_path.exists():
            with open(migration_log_path, 'r') as f:
                migration_log = json.load(f)
            
            for entry in migration_log:
                if (entry.get('action') == 'VERIFY' and 
                    entry.get('status') == 'success'):
                    migrated_schemas.append(entry.get('schema'))
        
        return migrated_schemas
    
    def check_archive_data(self, schema_name):
        """Check if archive data exists for schema."""
        archive_path = Path(self.archive_datadir) / schema_name
        
        try:
            result = run_command(['sudo', 'ls', str(archive_path)], capture_output=True, check=False)
            if result.returncode == 0:
                # Count .ibd files
                result = run_command(['sudo', 'find', str(archive_path), '-name', '*.ibd'], capture_output=True, check=False)
                if result.returncode == 0:
                    ibd_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                    return True, f"Found {ibd_count} .ibd files"
                else:
                    return False, "No .ibd files found"
            else:
                return False, "Archive directory not found"
        except Exception as e:
            return False, f"Check failed: {e}"
    
    def completely_rebuild_schema(self, schema_name):
        """Completely rebuild a schema from scratch."""
        primary_path = Path(self.primary_datadir) / schema_name
        archive_path = Path(self.archive_datadir) / schema_name
        
        try:
            if self.dry_run:
                archive_exists, archive_msg = self.check_archive_data(schema_name)
                self.log_action("REBUILD", schema_name, "success", f"DRY RUN: Would rebuild ({archive_msg})")
                return True
            
            # Step 1: Check if archive data exists
            archive_exists, archive_msg = self.check_archive_data(schema_name)
            if not archive_exists:
                self.log_action("REBUILD", schema_name, "error", f"No archive data: {archive_msg}")
                return False
            
            self.log_action("CHECK", schema_name, "info", archive_msg)
            
            # Step 2: Drop the broken database completely
            try:
                conn = create_connection()
                cursor = conn.cursor()
                cursor.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")
                cursor.close()
                conn.close()
                self.log_action("DROP", schema_name, "success", "Dropped broken database")
            except Exception as drop_error:
                self.log_action("DROP", schema_name, "error", f"Drop failed: {drop_error}")
                return False
            
            # Step 3: Remove any leftover files/links in MySQL directory  
            if primary_path.exists():
                run_command(['sudo', 'rm', '-rf', str(primary_path)])
                self.log_action("CLEANUP", schema_name, "success", "Removed leftover files")
            
            # Step 4: Copy archive data to primary location (NOT move, copy!)
            run_command(['sudo', 'cp', '-r', str(archive_path), str(primary_path)])
            run_command(['sudo', 'chown', '-R', 'mysql:mysql', str(primary_path)])
            run_command(['sudo', 'chmod', '-R', '750', str(primary_path)])
            self.log_action("COPY", schema_name, "success", "Copied archive data to primary")
            
            # Step 5: Try to import the schema structure back into MySQL
            # This is the critical step that fixes the metadata
            try:
                # Check if we can discover tables from the filesystem
                result = run_command(['sudo', 'find', str(primary_path), '-name', '*.frm'], capture_output=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    frm_files = result.stdout.strip().split('\n')
                    self.log_action("DISCOVER", schema_name, "info", f"Found {len(frm_files)} .frm files")
                    
                    # Create database
                    conn = create_connection()
                    cursor = conn.cursor()
                    cursor.execute(f"CREATE DATABASE `{schema_name}`")
                    cursor.close()
                    conn.close()
                    self.log_action("CREATE", schema_name, "success", "Created new database")
                    
                    # MySQL will discover the tables automatically when we connect to the database
                    conn = create_connection(schema_name)
                    cursor = conn.cursor()
                    cursor.execute("SHOW TABLES")
                    tables = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    if tables:
                        self.log_action("DISCOVER_TABLES", schema_name, "success", f"Discovered {len(tables)} tables")
                        return True
                    else:
                        self.log_action("DISCOVER_TABLES", schema_name, "error", "No tables discovered")
                        return False
                else:
                    self.log_action("DISCOVER", schema_name, "error", "No .frm files found in archive")
                    return False
                    
            except Exception as import_error:
                self.log_action("IMPORT", schema_name, "error", f"Import failed: {import_error}")
                return False
                
        except Exception as e:
            self.log_action("REBUILD", schema_name, "error", f"Rebuild failed: {e}")
            return False
    
    def test_schema_access(self, schema_name):
        """Test if rebuilt schema has full data access."""
        try:
            conn = create_connection(schema_name)
            cursor = conn.cursor()
            
            # Get first table
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            if not tables:
                cursor.close()
                conn.close()
                return True, "Empty schema"
            
            table_name = tables[0][0]
            
            # Try to access data
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            return True, f"Full access: {len(tables)} tables, {count} rows in sample"
            
        except Exception as e:
            return False, str(e)
    
    def run_rebuild(self):
        """Run the complete schema rebuild."""
        print("🔨 COMPLETELY REBUILDING SCHEMAS FROM ARCHIVE DATA")
        print("=" * 80)
        print("This is the nuclear option - completely rebuilds schemas to fix")
        print("severely corrupted InnoDB metadata that can't be fixed by file moves.")
        print()
        
        # Get list of migrated schemas
        migrated_schemas = self.get_migrated_schemas()
        print(f"📋 Found {len(migrated_schemas)} migrated schemas to rebuild")
        
        if not migrated_schemas:
            print("❌ No migrated schemas found in log")
            return
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")
        else:
            print("⚠️  WARNING: This will COMPLETELY REBUILD all schemas!")
            print("   • DROP all broken databases")
            print("   • Copy archive data to primary storage")  
            print("   • Recreate databases from filesystem data")
            print("   • This is IRREVERSIBLE but should fix the tablespace issues")
            print()
            confirm = input(f"Proceed with COMPLETE REBUILD of {len(migrated_schemas)} schemas? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Rebuild cancelled by user")
                return
        
        successful_rebuilds = 0
        failed_rebuilds = 0
        
        # Start with first 3 schemas as a test
        test_schemas = migrated_schemas[:3]
        print(f"\n🔨 Starting with first {len(test_schemas)} schemas as a test...")
        
        for i, schema_name in enumerate(test_schemas, 1):
            print(f"\n[{i}/{len(test_schemas)}] Rebuilding: {schema_name}")
            
            if self.completely_rebuild_schema(schema_name):
                successful_rebuilds += 1
            else:
                failed_rebuilds += 1
        
        # Test the rebuilt schemas
        if not self.dry_run:
            print(f"\n🧪 Testing access to rebuilt schemas...")
            working_count = 0
            
            for schema_name in test_schemas:
                success, message = self.test_schema_access(schema_name)
                status = "✅" if success else "❌"
                print(f"   {status} {schema_name}: {message}")
                if success:
                    working_count += 1
            
            print(f"\n📊 TEST RESULTS:")
            print(f"   ✅ Rebuilt: {successful_rebuilds}/{len(test_schemas)}")
            print(f"   ✅ Access working: {working_count}/{len(test_schemas)}")
            
            if working_count > 0:
                print(f"\n🎉 REBUILD WORKING! Continue with remaining schemas? (y/N): ", end="")
                continue_choice = input().strip().lower()
                if continue_choice == 'y':
                    # Rebuild remaining schemas
                    remaining_schemas = migrated_schemas[3:]
                    for i, schema_name in enumerate(remaining_schemas, 4):
                        print(f"\n[{i}/{len(migrated_schemas)}] Rebuilding: {schema_name}")
                        if self.completely_rebuild_schema(schema_name):
                            successful_rebuilds += 1
                        else:
                            failed_rebuilds += 1
                else:
                    print("❌ Stopped at user request")
            else:
                print(f"\n❌ Rebuild test failed. Stopping here.")
        
        # Save log
        log_path = Path(self.log_file)
        with open(log_path, 'w') as f:
            json.dump(self.rebuild_log, f, indent=2)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   ✅ Successfully rebuilt: {successful_rebuilds}")
        print(f"   ❌ Failed: {failed_rebuilds}")
        print(f"   📄 Log saved: {log_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Completely rebuild schemas from archive data")
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--live', action='store_true', help='Live rebuild mode')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.live:
        print("Please specify --dry-run or --live")
        return
    
    rebuilder = SchemaRebuilder(dry_run=args.dry_run)
    rebuilder.run_rebuild()

if __name__ == "__main__":
    main()