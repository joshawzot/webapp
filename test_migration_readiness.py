#!/usr/bin/env python3
"""
Test migration readiness and verify all components are working
"""

import subprocess
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

def run_cmd(cmd, capture_output=True):
    """Run command and return result."""
    try:
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
    print("🔍 MIGRATION READINESS TEST")
    print("=" * 60)
    
    # Test 1: MySQL connectivity
    print("1️⃣ Testing MySQL connectivity...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SELECT VERSION();'])
    if success:
        mysql_version = output.split('\n')[1] if '\n' in output else output
        print(f"   ✅ MySQL connected: {mysql_version}")
    else:
        print(f"   ❌ MySQL connection failed: {output}")
        return
    
    # Test 2: Check current schema count
    print("\n2️⃣ Checking current schema status...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if success:
        all_databases = output.split('\n')[1:]  # Skip header
        user_databases = [db for db in all_databases 
                         if db not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        print(f"   📊 Total user schemas: {len(user_databases)}")
        
        # Count schemas by age
        old_schemas = []
        new_schemas = []
        no_timestamp = []
        cutoff_date = datetime.now() - timedelta(days=60)
        
        for schema in user_databases:
            creation_time = extract_timestamp_from_name(schema)
            if creation_time:
                if creation_time < cutoff_date:
                    age_days = (datetime.now() - creation_time).days
                    old_schemas.append((schema, age_days))
                else:
                    new_schemas.append(schema)
            else:
                no_timestamp.append(schema)
        
        print(f"   📅 Schemas >60 days old: {len(old_schemas)} (will be migrated)")
        print(f"   📅 Schemas <60 days old: {len(new_schemas)} (will stay)")
        print(f"   📅 Schemas without timestamp: {len(no_timestamp)} (will stay)")
        
        if len(old_schemas) > 0:
            oldest = max(old_schemas, key=lambda x: x[1])
            print(f"   📈 Oldest schema: {oldest[0]} ({oldest[1]} days)")
            
            print(f"   📝 Sample schemas to migrate:")
            for schema, age in sorted(old_schemas, key=lambda x: x[1], reverse=True)[:5]:
                print(f"      • {schema} ({age} days)")
    else:
        print(f"   ❌ Could not get database list: {output}")
        return
    
    # Test 3: Check storage locations
    print("\n3️⃣ Checking storage locations...")
    
    primary_storage = "/var/lib/mysql"
    archive_storage = "/local/mysql/data"
    backup_directory = "/local/mysql_migration_backups"
    
    # Check primary storage
    if os.path.exists(primary_storage):
        success, output = run_cmd(['df', '-h', primary_storage])
        if success:
            df_line = output.split('\n')[1].split()
            device = df_line[0]
            available = df_line[3]
            print(f"   ✅ Primary storage: {device} ({available} available)")
        else:
            print(f"   ✅ Primary storage: {primary_storage} (exists)")
    else:
        print(f"   ❌ Primary storage not found: {primary_storage}")
    
    # Check archive storage
    if os.path.exists(archive_storage):
        success, output = run_cmd(['df', '-h', archive_storage])
        if success:
            df_line = output.split('\n')[1].split()
            device = df_line[0]
            available = df_line[3]
            print(f"   ✅ Archive storage: {device} ({available} available)")
        else:
            print(f"   ✅ Archive storage: {archive_storage} (exists)")
    else:
        print(f"   ⚠️  Archive storage not found: {archive_storage}")
        print(f"      Will be created during migration")
    
    # Check backup directory
    if os.path.exists(backup_directory):
        print(f"   ✅ Backup directory exists: {backup_directory}")
    else:
        print(f"   ⚠️  Backup directory not found: {backup_directory}")
        print(f"      Will be created during migration")
    
    # Test 4: Check permissions
    print("\n4️⃣ Checking permissions...")
    
    # Check if running as root
    if os.geteuid() == 0:
        print(f"   ✅ Running as root (required for migration)")
    else:
        print(f"   ❌ Not running as root")
        print(f"      Migration scripts must be run with sudo")
    
    # Test 5: Check mysqldump
    print("\n5️⃣ Testing mysqldump...")
    success, output = run_cmd(['mysqldump', '--version'])
    if success:
        print(f"   ✅ mysqldump available: {output}")
    else:
        print(f"   ❌ mysqldump not available: {output}")
    
    # Test 6: Test storage detection system
    print("\n6️⃣ Testing storage detection...")
    try:
        sys.path.append('/home/admin2/webapp_2')
        from dual_storage_db_operations import DualStorageManager
        
        dual_storage = DualStorageManager()
        test_schema = user_databases[0] if user_databases else "test"
        location = dual_storage.get_schema_location(test_schema)
        print(f"   ✅ Storage detection working: {test_schema} → {location}")
    except Exception as e:
        print(f"   ❌ Storage detection error: {e}")
    
    # Summary
    print(f"\n📊 READINESS SUMMARY:")
    
    if len(old_schemas) > 0:
        print(f"   🎯 Ready to migrate {len(old_schemas)} schemas")
        print(f"   💾 Estimated backup size: ~{len(old_schemas) * 50} MB")
        print(f"   ⏱️  Estimated migration time: ~{len(old_schemas) * 30 // 60} minutes")
        
        print(f"\n🚀 TO START MIGRATION:")
        print(f"   sudo python3 safe_mysql_migration.py")
        
        print(f"\n🔄 IF ROLLBACK NEEDED:")
        print(f"   sudo python3 rollback_migration.py")
        
        print(f"\n✅ SAFETY FEATURES:")
        print(f"   • SQL backups created before migration")
        print(f"   • MySQL-native methods (no file corruption)")
        print(f"   • Rollback capability available")
        print(f"   • Migration log for tracking")
    else:
        print(f"   ✅ No schemas need migration (all are <60 days old)")
        print(f"   💡 All schemas will remain on primary storage")

if __name__ == "__main__":
    main()