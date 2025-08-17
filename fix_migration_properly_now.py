#!/usr/bin/env python3
"""
PROPERLY fix the migration by correctly moving files back and removing symbolic links
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
        print(f"Running: {' '.join(cmd)}")
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

class ProperMigrationFixer:
    """Properly fix the migration by correctly handling files and symbolic links."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.primary_datadir = "/var/lib/mysql"
        self.archive_datadir = "/local/mysql/data"
        self.log_file = f"proper_fix_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.fix_log = []
        
    def log_action(self, action, schema_name, status, message):
        """Log fix actions."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "schema": schema_name,
            "status": status,
            "message": message
        }
        self.fix_log.append(entry)
        print(f"[{action}] {schema_name}: {status} - {message}")
    
    def get_migrated_schemas(self):
        """Get list of all migrated schemas from the log."""
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
    
    def fix_single_schema(self, schema_name):
        """Properly fix a single schema."""
        primary_path = Path(self.primary_datadir) / schema_name
        archive_path = Path(self.archive_datadir) / schema_name
        
        try:
            if self.dry_run:
                self.log_action("FIX", schema_name, "success", f"DRY RUN: Would fix {schema_name}")
                return True
            
            # Step 1: Check current state
            result = run_command(['sudo', 'ls', '-la', str(primary_path)], capture_output=True, check=False)
            is_symlink = 'lrwx' in result.stdout if result.returncode == 0 else False
            
            if is_symlink:
                self.log_action("CHECK", schema_name, "info", "Found symbolic link")
                
                # Step 2: Check if archive directory exists
                result = run_command(['sudo', 'ls', '-d', str(archive_path)], capture_output=True, check=False)
                if result.returncode != 0:
                    self.log_action("FIX", schema_name, "error", f"Archive directory not found: {archive_path}")
                    return False
                
                # Step 3: Remove the symbolic link
                run_command(['sudo', 'rm', str(primary_path)])
                self.log_action("REMOVE_LINK", schema_name, "success", "Removed symbolic link")
                
                # Step 4: Move files from archive back to primary
                run_command(['sudo', 'mv', str(archive_path), str(primary_path)])
                self.log_action("MOVE_FILES", schema_name, "success", f"Moved {archive_path} -> {primary_path}")
                
                # Step 5: Set correct ownership and permissions
                run_command(['sudo', 'chown', '-R', 'mysql:mysql', str(primary_path)])
                run_command(['sudo', 'chmod', '-R', '750', str(primary_path)])
                self.log_action("PERMISSIONS", schema_name, "success", "Set correct permissions")
                
                return True
            else:
                self.log_action("CHECK", schema_name, "info", "Not a symbolic link - may already be fixed")
                return True
                
        except Exception as e:
            self.log_action("FIX", schema_name, "error", f"Fix failed: {e}")
            return False
    
    def restart_mysql(self):
        """Restart MySQL service to clear any cached metadata."""
        try:
            if self.dry_run:
                print("🔄 DRY RUN: Would restart MySQL service")
                return True
            
            print("🔄 Restarting MySQL service...")
            
            # Stop MySQL
            run_command(['sudo', 'systemctl', 'stop', 'mysql'])
            print("⏹️  MySQL stopped")
            
            # Wait a moment
            time.sleep(3)
            
            # Start MySQL
            run_command(['sudo', 'systemctl', 'start', 'mysql'])
            print("▶️  MySQL started")
            
            # Wait for MySQL to fully start
            time.sleep(5)
            
            return True
            
        except Exception as e:
            print(f"❌ MySQL restart failed: {e}")
            return False
    
    def test_schema_access(self, schema_name):
        """Test if schema has full data access."""
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
            return True, f"Access OK: {count} rows"
            
        except Exception as e:
            return False, str(e)
    
    def run_proper_fix(self):
        """Run the proper fix."""
        print("🔧 PROPERLY FIXING MIGRATION ISSUES")
        print("=" * 80)
        
        # Get list of migrated schemas
        migrated_schemas = self.get_migrated_schemas()
        print(f"📋 Found {len(migrated_schemas)} migrated schemas to fix")
        
        if not migrated_schemas:
            print("❌ No migrated schemas found in log")
            return
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")
        else:
            print("⚠️  This will:")
            print("   • Remove symbolic links")
            print("   • Move schema files from archive back to primary storage") 
            print("   • Restart MySQL service")
            print("   • Test schema access")
            print()
            confirm = input(f"Proceed with proper fix for {len(migrated_schemas)} schemas? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Fix cancelled by user")
                return
        
        successful_fixes = 0
        failed_fixes = 0
        
        # Fix first 5 schemas as a test
        test_schemas = migrated_schemas[:5]
        print(f"\n🔧 Starting with first {len(test_schemas)} schemas as a test...")
        
        for i, schema_name in enumerate(test_schemas, 1):
            print(f"\n[{i}/{len(test_schemas)}] Fixing: {schema_name}")
            
            if self.fix_single_schema(schema_name):
                successful_fixes += 1
            else:
                failed_fixes += 1
        
        # Restart MySQL after file changes
        if not self.dry_run:
            if not self.restart_mysql():
                print("❌ MySQL restart failed - fix may not work")
                return
        
        # Test the fixed schemas
        print(f"\n🧪 Testing access to fixed schemas...")
        working_count = 0
        
        for schema_name in test_schemas:
            success, message = self.test_schema_access(schema_name)
            status = "✅" if success else "❌"
            print(f"   {status} {schema_name}: {message}")
            if success:
                working_count += 1
        
        print(f"\n📊 TEST RESULTS:")
        print(f"   ✅ Files moved: {successful_fixes}/{len(test_schemas)}")
        print(f"   ✅ Access working: {working_count}/{len(test_schemas)}")
        
        if working_count == len(test_schemas):
            print(f"\n🎉 TEST SUCCESSFUL! Proceeding with remaining schemas...")
            
            # Fix remaining schemas
            remaining_schemas = migrated_schemas[5:]
            for i, schema_name in enumerate(remaining_schemas, 6):
                print(f"\n[{i}/{len(migrated_schemas)}] Fixing: {schema_name}")
                if self.fix_single_schema(schema_name):
                    successful_fixes += 1
                else:
                    failed_fixes += 1
        else:
            print(f"\n⚠️  Test partially failed. Check results before proceeding.")
        
        # Save log
        log_path = Path(self.log_file)
        with open(log_path, 'w') as f:
            json.dump(self.fix_log, f, indent=2)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   ✅ Successfully fixed: {successful_fixes}")
        print(f"   ❌ Failed: {failed_fixes}")
        print(f"   📄 Log saved: {log_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Properly fix migration by moving files back and removing symbolic links")
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--live', action='store_true', help='Live fix mode')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.live:
        print("Please specify --dry-run or --live")
        return
    
    fixer = ProperMigrationFixer(dry_run=args.dry_run)
    fixer.run_proper_fix()

if __name__ == "__main__":
    main()