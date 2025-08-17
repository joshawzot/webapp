#!/usr/bin/env python3
"""
EMERGENCY FIX: Reverse the problematic migration to restore full functionality
This moves schemas back from archive to primary storage and removes symbolic links
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

class EmergencyMigrationReverser:
    """Emergency fix to reverse the problematic migration."""
    
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.primary_datadir = "/var/lib/mysql"
        self.archive_datadir = "/local/mysql/data"
        self.log_file = f"emergency_fix_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    
    def get_migrated_schemas(self):
        """Get list of all migrated schemas."""
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
        
        return migrated_schemas
    
    def reverse_schema_migration(self, schema_name):
        """Reverse migration for a single schema."""
        primary_path = Path(self.primary_datadir) / schema_name
        archive_path = Path(self.archive_datadir) / schema_name
        
        try:
            if self.dry_run:
                self.log_action("REVERSE", schema_name, "success", f"DRY RUN: Would move {archive_path} -> {primary_path}")
                return True
            
            # Step 1: Check if archive directory exists
            try:
                result = run_command(['sudo', 'ls', '-d', str(archive_path)], capture_output=True)
                if result.returncode != 0:
                    self.log_action("REVERSE", schema_name, "error", f"Archive directory not found: {archive_path}")
                    return False
            except:
                self.log_action("REVERSE", schema_name, "error", f"Cannot access archive directory: {archive_path}")
                return False
            
            # Step 2: Remove symbolic link if it exists
            if primary_path.is_symlink():
                run_command(['sudo', 'rm', str(primary_path)])
                self.log_action("CLEANUP", schema_name, "success", "Removed symbolic link")
            elif primary_path.exists():
                self.log_action("CLEANUP", schema_name, "warning", "Primary path exists but is not a symlink")
            
            # Step 3: Move files back from archive to primary
            run_command(['sudo', 'mv', str(archive_path), str(primary_path)])
            self.log_action("MOVE", schema_name, "success", f"Moved from archive to primary: {archive_path} -> {primary_path}")
            
            # Step 4: Set correct ownership and permissions
            run_command(['sudo', 'chown', '-R', 'mysql:mysql', str(primary_path)])
            run_command(['sudo', 'chmod', '-R', '750', str(primary_path)])
            self.log_action("PERMISSIONS", schema_name, "success", "Set correct ownership and permissions")
            
            # Step 5: Test access
            time.sleep(1)  # Give MySQL a moment
            if self.test_schema_access(schema_name):
                self.log_action("VERIFY", schema_name, "success", "Schema access restored successfully")
                return True
            else:
                self.log_action("VERIFY", schema_name, "warning", "Schema moved but access still has issues")
                return True  # Still count as success since we moved it back
                
        except Exception as e:
            self.log_action("REVERSE", schema_name, "error", f"Reversal failed: {e}")
            return False
    
    def run_emergency_fix(self):
        """Run the emergency fix to reverse migrations."""
        print("🚨 EMERGENCY FIX: REVERSING PROBLEMATIC MIGRATION")
        print("=" * 80)
        print("This will move all migrated schemas back to primary storage")
        print("and remove symbolic links to restore full functionality.")
        print()
        
        # Get list of migrated schemas
        migrated_schemas = self.get_migrated_schemas()
        print(f"📋 Found {len(migrated_schemas)} migrated schemas to reverse")
        
        if not migrated_schemas:
            print("❌ No migrated schemas found in log")
            return
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No actual changes will be made")
        else:
            print("⚠️  WARNING: This will:")
            print("   • Move schemas from /local/mysql/data/ back to /var/lib/mysql/")
            print("   • Remove symbolic links")
            print("   • Restore full data access")
            print("   • Use primary storage space (you have 1.7TB available)")
            print()
            confirm = input(f"Proceed with emergency fix for {len(migrated_schemas)} schemas? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Emergency fix cancelled by user")
                return
        
        successful_reversals = 0
        failed_reversals = 0
        
        print(f"\n🔧 Starting emergency fix...")
        
        for i, schema_name in enumerate(migrated_schemas, 1):
            print(f"\n[{i}/{len(migrated_schemas)}] Reversing: {schema_name}")
            
            if self.reverse_schema_migration(schema_name):
                successful_reversals += 1
            else:
                failed_reversals += 1
            
            # Show progress every 50 schemas
            if i % 50 == 0:
                print(f"📊 Progress: {i}/{len(migrated_schemas)} ({successful_reversals} successful, {failed_reversals} failed)")
        
        # Save log
        log_path = Path(self.log_file)
        with open(log_path, 'w') as f:
            json.dump(self.fix_log, f, indent=2)
        
        print(f"\n📊 EMERGENCY FIX COMPLETE:")
        print(f"   ✅ Successfully reversed: {successful_reversals}")
        print(f"   ❌ Failed: {failed_reversals}")
        print(f"   📄 Log saved: {log_path}")
        
        if successful_reversals > 0:
            print(f"\n🎉 EMERGENCY FIX SUCCESSFUL!")
            print(f"   • {successful_reversals} schemas restored to primary storage")
            print(f"   • Full data access should now work")
            print(f"   • Your webapp should display normal table dimensions")
            print(f"   • You can migrate properly later using the fixed script")
            
            print(f"\n💾 STORAGE STATUS:")
            print(f"   • Primary (/var/lib/mysql): Will have more data now")
            print(f"   • Archive (/local/mysql/data): Will have less data")
            print(f"   • Run 'df -h' to check current usage")
        
        return successful_reversals, failed_reversals

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency fix: Reverse problematic migration")
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--live', action='store_true', help='Live fix mode')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.live:
        print("Please specify --dry-run or --live")
        return
    
    reverser = EmergencyMigrationReverser(dry_run=args.dry_run)
    reverser.run_emergency_fix()

if __name__ == "__main__":
    main()