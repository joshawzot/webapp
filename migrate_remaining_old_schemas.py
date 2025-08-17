#!/usr/bin/env python3
"""
Migrate the remaining 34 old schemas (>60 days) from primary storage to SDA1 via symlinks.
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import get_schema_storage_type, extract_timestamp_from_schema_name
from datetime import datetime, timedelta
import mysql.connector
import subprocess
import os

def get_unmigrated_old_schemas():
    """Get list of old schemas that need migration."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        cursor = connection.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall() 
                    if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        cursor.close()
        connection.close()
        
        cutoff_date = datetime.now() - timedelta(days=60)
        unmigrated = []
        
        for schema in databases:
            timestamp = extract_timestamp_from_schema_name(schema)
            if timestamp and (datetime.now() - timestamp).days >= 60:
                storage_info = get_schema_storage_type(schema)
                if storage_info['type'] == 'primary':
                    unmigrated.append({
                        'name': schema,
                        'age_days': (datetime.now() - timestamp).days,
                        'timestamp': timestamp
                    })
        
        return unmigrated
        
    except Exception as e:
        print(f"Error getting schemas: {e}")
        return []

def migrate_schema_to_sda1(schema_name):
    """Migrate a single schema from primary to SDA1 via symlink."""
    print(f"🔄 Migrating: {schema_name}")
    
    primary_path = f"/app/mysql/{schema_name}"
    archive_path = f"/local/mysql_old/{schema_name}"
    
    try:
        # Step 1: Stop MySQL to ensure no active connections
        print("   ⏸️  Stopping MySQL...")
        subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], check=True)
        
        # Step 2: Move the schema directory to SDA1
        print("   📦 Moving to SDA1...")
        subprocess.run(['sudo', 'mv', primary_path, archive_path], check=True)
        
        # Step 3: Create symlink from primary to archive
        print("   🔗 Creating symlink...")
        subprocess.run(['sudo', 'ln', '-s', archive_path, primary_path], check=True)
        
        # Step 4: Fix ownership
        print("   👤 Fixing ownership...")
        subprocess.run(['sudo', 'chown', 'mysql:mysql', primary_path], check=True)
        subprocess.run(['sudo', 'chown', '-R', 'mysql:mysql', archive_path], check=True)
        
        # Step 5: Start MySQL
        print("   ▶️  Starting MySQL...")
        subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=True)
        
        # Step 6: Verify the migration
        print("   ✅ Verifying...")
        storage_info = get_schema_storage_type(schema_name)
        if storage_info['type'] == 'archive_with_direct_access':
            print(f"   🎯 SUCCESS: {schema_name} now on SDA1!")
            return True
        else:
            print(f"   ❌ FAILED: Migration verification failed")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        # Try to restart MySQL if it's stopped
        try:
            subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=True)
        except:
            pass
        return False

def migrate_all_remaining():
    """Migrate all remaining old schemas to SDA1."""
    print("🚀 MIGRATING REMAINING OLD SCHEMAS TO SDA1")
    print("=" * 60)
    
    unmigrated = get_unmigrated_old_schemas()
    
    if not unmigrated:
        print("🎉 No schemas need migration - all are properly allocated!")
        return
    
    print(f"📋 Found {len(unmigrated)} schemas to migrate")
    print(f"⚠️  This process will restart MySQL multiple times")
    
    # Ask for confirmation
    response = input(f"\n🤔 Proceed with migrating {len(unmigrated)} schemas? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled")
        return
    
    successful = 0
    failed = 0
    
    for i, schema in enumerate(unmigrated, 1):
        print(f"\n[{i}/{len(unmigrated)}] Processing: {schema['name']}")
        print(f"   Age: {schema['age_days']} days")
        
        if migrate_schema_to_sda1(schema['name']):
            successful += 1
        else:
            failed += 1
            print(f"   ⚠️  Continuing with next schema...")
    
    # Final summary
    print(f"\n📊 MIGRATION COMPLETE!")
    print(f"=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {successful/(successful+failed)*100:.1f}%")
    
    if successful > 0:
        print(f"\n🎉 {successful} schemas now have direct SDA1 access!")
        print(f"📦 Your webapp will show them as 'Archive Storage (Direct Access)'")

if __name__ == "__main__":
    migrate_all_remaining()