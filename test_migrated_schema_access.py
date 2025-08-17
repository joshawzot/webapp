#!/usr/bin/env python3
"""
Test script to verify dual storage access for migrated schemas
"""

import sys
import os
sys.path.append('/home/admin2/webapp_2')

from dual_storage_db_operations import DualStorageManager
from db_operations import create_connection
import mysql.connector

def test_migrated_schema_access():
    """Test accessing a migrated schema using dual storage detection"""
    
    schema_name = "MaxZhang_Cullinan_2428_KG2_H09_310_20250417102127"
    print(f"🔍 Testing access to migrated schema: {schema_name}")
    print("=" * 60)
    
    # Test 1: Direct MySQL connection (should fail)
    print("\n1️⃣ Testing direct MySQL connection (expected to fail):")
    try:
        conn = create_connection(schema_name)
        print("❌ Unexpected: Direct connection succeeded (schema might be restored)")
        conn.close()
    except mysql.connector.Error as e:
        print(f"✅ Expected: Direct connection failed - {e}")
    
    # Test 2: Dual storage detection
    print("\n2️⃣ Testing dual storage detection:")
    try:
        dual_storage = DualStorageManager()
        location = dual_storage.get_schema_location(schema_name)
        print(f"📍 Schema location: {location}")
        
        storage_info = dual_storage.get_storage_info(schema_name)
        if storage_info:
            print(f"💾 Storage info:")
            for key, value in storage_info.items():
                print(f"   {key}: {value}")
        else:
            print("❌ No storage info found")
            
    except Exception as e:
        print(f"❌ Error in dual storage detection: {e}")
    
    # Test 3: Check backup existence
    print("\n3️⃣ Checking backup file:")
    backup_path = f"/local/mysql_migration_backups/{schema_name}.sql"
    if os.path.exists(backup_path):
        size = os.path.getsize(backup_path)
        print(f"✅ Backup exists: {backup_path}")
        print(f"📦 Backup size: {size / (1024*1024):.1f} MB")
    else:
        print(f"❌ Backup not found: {backup_path}")
    
    # Test 4: Archive directory check
    print("\n4️⃣ Checking archive directory:")
    archive_path = f"/local/mysql/data/{schema_name}"
    try:
        if os.path.exists(archive_path):
            print(f"✅ Archive directory exists: {archive_path}")
            files = os.listdir(archive_path)
            print(f"📁 Files in archive: {len(files)} files")
        else:
            print(f"❌ Archive directory not found: {archive_path}")
    except PermissionError:
        print(f"🔒 Permission denied accessing: {archive_path}")
        print("   (This is normal - archive is owned by mysql user)")

def show_restoration_commands():
    """Show commands to restore the schema"""
    schema_name = "MaxZhang_Cullinan_2428_KG2_H09_310_20250417102127"
    
    print("\n" + "=" * 60)
    print("🔧 RESTORATION OPTIONS:")
    print("=" * 60)
    
    print(f"\n📤 To restore {schema_name} to primary storage:")
    print(f"sudo mysql -u root < /local/mysql_migration_backups/{schema_name}.sql")
    
    print(f"\n📋 To list all migrated schemas:")
    print("ls -la /local/mysql_migration_backups/ | head -20")
    
    print(f"\n📊 To check migration statistics:")
    print("grep -c '\"status\": \"success\"' safe_migration_log_20250804_010903.json")

if __name__ == "__main__":
    test_migrated_schema_access()
    show_restoration_commands()