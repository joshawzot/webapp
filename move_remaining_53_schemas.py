#!/usr/bin/env python3
"""
Move the remaining 53 schemas from archive to primary
These were missed by the nuclear fix
Must be run with sudo: sudo python3 move_remaining_53_schemas.py
"""

import subprocess
import os
import sys
import time

def run_cmd(cmd, capture_output=True):
    """Run command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
        return True, result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run with sudo!")
        print("Usage: sudo python3 move_remaining_53_schemas.py")
        sys.exit(1)
    
    print("🔧 MOVING REMAINING 53 SCHEMAS")
    print("=" * 60)
    print("These schemas were missed by the nuclear fix")
    print()
    
    # Check what's actually in archive
    success, output = run_cmd(['sudo', 'ls', '/local/mysql/data/'])
    if not success:
        print(f"❌ Cannot access archive: {output}")
        return
    
    archive_dirs = output.split('\n') if output else []
    maxzhang_dirs = [d for d in archive_dirs if 'MaxZhang' in d]
    
    print(f"📦 Found {len(maxzhang_dirs)} MaxZhang schemas in archive:")
    for schema in maxzhang_dirs[:10]:  # Show first 10
        print(f"   📁 {schema}")
    if len(maxzhang_dirs) > 10:
        print(f"   📁 ... and {len(maxzhang_dirs) - 10} more")
    print()
    
    # Check the specific schema the user mentioned
    target_schema = "MaxZhang_Cullinan_2331_JH1_I03_KPI_20250327082452"
    if target_schema in maxzhang_dirs:
        print(f"🎯 Found target schema: {target_schema}")
    else:
        print(f"⚠️  Target schema not found in archive")
    print()
    
    if len(maxzhang_dirs) == 0:
        print("✅ No schemas found in archive!")
        return
    
    confirm = input(f"Move {len(maxzhang_dirs)} schemas from archive to primary? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Cancelled by user")
        return
    
    print(f"\n🚚 Moving {len(maxzhang_dirs)} schemas...")
    successful_moves = 0
    failed_moves = 0
    
    # Stop MySQL for file operations
    print("⏹️  Stopping MySQL...")
    success, output = run_cmd(['systemctl', 'stop', 'mysql'])
    if not success:
        print(f"❌ Failed to stop MySQL: {output}")
        return
    
    time.sleep(3)
    
    for i, schema in enumerate(maxzhang_dirs, 1):
        print(f"[{i}/{len(maxzhang_dirs)}] Moving: {schema}")
        
        archive_path = f"/local/mysql/data/{schema}"
        primary_path = f"/var/lib/mysql/{schema}"
        backup_path = f"/local/mysql_corrupted_backup/{schema}"
        
        try:
            # Create backup directory if needed
            run_cmd(['mkdir', '-p', '/local/mysql_corrupted_backup'])
            
            # Backup to safety location
            success, output = run_cmd(['cp', '-r', archive_path, backup_path])
            if not success:
                print(f"   ❌ Backup failed: {output}")
                failed_moves += 1
                continue
            
            # Check if already exists in primary (conflict resolution)
            success, output = run_cmd(['ls', '-d', primary_path], capture_output=True)
            if success:
                print(f"   ⚠️  Removing existing primary copy...")
                success, output = run_cmd(['rm', '-rf', primary_path])
                if not success:
                    print(f"   ❌ Cannot remove existing: {output}")
                    failed_moves += 1
                    continue
            
            # Move from archive to primary
            success, output = run_cmd(['mv', archive_path, primary_path])
            if not success:
                print(f"   ❌ Move failed: {output}")
                failed_moves += 1
                continue
            
            # Set correct permissions
            success, output = run_cmd(['chown', '-R', 'mysql:mysql', primary_path])
            if not success:
                print(f"   ❌ Permission setting failed: {output}")
                failed_moves += 1
                continue
            
            print(f"   ✅ Moved successfully")
            successful_moves += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed_moves += 1
    
    # Start MySQL
    print("\n▶️  Starting MySQL...")
    success, output = run_cmd(['systemctl', 'start', 'mysql'])
    if not success:
        print(f"❌ Failed to start MySQL: {output}")
        return
    
    time.sleep(5)
    
    # Create database entries for moved schemas
    print("\n🔧 Creating database entries...")
    database_successful = 0
    
    for schema in maxzhang_dirs[:successful_moves]:
        success, output = run_cmd(['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema}`;'])
        if success:
            database_successful += 1
            
            # Quick table test
            success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{schema}`; SHOW TABLES;'])
            if success:
                table_lines = output.split('\n')[1:] if '\n' in output else []
                table_count = len([t for t in table_lines if t.strip()])
                if table_count > 0:
                    print(f"   🎉 {schema}: {table_count} tables")
                else:
                    print(f"   ⚠️  {schema}: database created, tables pending")
        else:
            print(f"   ❌ Database creation failed for {schema}")
    
    print(f"\n📊 OPERATION COMPLETE:")
    print(f"   ✅ Files moved: {successful_moves}/{len(maxzhang_dirs)}")
    print(f"   ❌ Failed moves: {failed_moves}/{len(maxzhang_dirs)}")
    print(f"   🗄️  Database entries: {database_successful}/{successful_moves}")
    
    if successful_moves > 0:
        print(f"\n🎉 SUCCESS!")
        print(f"   • {successful_moves} schemas moved to primary storage")
        print(f"   • Refresh your webapp to see the change")
        print(f"   • Your schema should now show 'Primary Storage'")

if __name__ == "__main__":
    main()