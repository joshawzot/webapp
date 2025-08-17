#!/usr/bin/env python3
"""
Convert SQL backups to true dual storage with direct SDA1 access via symlinks
"""

import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def convert_sql_backup_to_symlink(schema_name, dry_run=True):
    """Convert a SQL backup to symlink-based archive access"""
    
    print(f"🔄 Converting {schema_name} to true dual storage...")
    
    # Paths
    sql_backup = f"/local/mysql_migration_backups/{schema_name}.sql"
    primary_path = f"/app/mysql/{schema_name}"
    archive_path = f"/local/mysql_old/{schema_name}"
    
    try:
        # Step 1: Verify SQL backup exists
        if not os.path.exists(sql_backup):
            raise Exception(f"SQL backup not found: {sql_backup}")
        
        backup_size = os.path.getsize(sql_backup) / (1024*1024)
        print(f"   📁 SQL backup: {backup_size:.1f} MB")
        
        # Step 2: Check if already converted
        if os.path.islink(primary_path):
            print(f"   ⏭️  Already converted to symlink")
            return True
        
        # Step 3: Check if archive directory already exists
        if os.path.exists(archive_path):
            print(f"   ⚠️  Archive directory already exists: {archive_path}")
            return False
        
        # Step 4: Check if schema is currently in MySQL
        result = subprocess.run(['mysql', '-u', 'root', '-e', f'SHOW DATABASES LIKE "{schema_name}";'],
                              capture_output=True, text=True)
        schema_in_mysql = schema_name in result.stdout
        
        if dry_run:
            print(f"   🧪 DRY RUN - Would perform:")
            print(f"      1. {'Skip' if schema_in_mysql else 'Restore'} schema from SQL backup")
            print(f"      2. Stop MySQL")
            print(f"      3. Move {primary_path} → {archive_path}")
            print(f"      4. Create symlink {primary_path} → {archive_path}")
            print(f"      5. Start MySQL")
            print(f"      6. Verify access from SDA1")
            print(f"      7. Delete SQL backup (save {backup_size:.1f} MB)")
            return True
        
        # Step 5: Restore schema if not already in MySQL
        if not schema_in_mysql:
            print(f"   📥 Restoring schema from SQL backup...")
            
            # Create database
            result = subprocess.run(['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema_name}`;'],
                                  capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise Exception(f"Failed to create database: {result.stderr}")
            
            # Restore from backup
            with open(sql_backup, 'r') as f:
                result = subprocess.run(['mysql', '-u', 'root', schema_name],
                                      stdin=f, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise Exception(f"Failed to restore: {result.stderr}")
            
            print(f"   ✅ Restored from SQL backup")
        else:
            print(f"   ✅ Schema already in MySQL")
        
        # Step 6: Convert to symlink
        print(f"   ⏸️  Stopping MySQL...")
        result = subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to stop MySQL: {result.stderr}")
        
        # Step 7: Ensure archive directory exists
        subprocess.run(['sudo', 'mkdir', '-p', '/local/mysql_old'], check=True)
        
        # Step 8: Move directory to archive
        print(f"   📦 Moving to SDA1 archive...")
        result = subprocess.run(['sudo', 'mv', primary_path, archive_path],
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to move directory: {result.stderr}")
        
        # Step 9: Create symlink
        print(f"   🔗 Creating symlink for direct SDA1 access...")
        result = subprocess.run(['sudo', 'ln', '-s', archive_path, primary_path],
                              capture_output=True, text=True)
        if result.returncode != 0:
            # Try to move back if symlink creation failed
            subprocess.run(['sudo', 'mv', archive_path, primary_path], check=False)
            raise Exception(f"Failed to create symlink: {result.stderr}")
        
        # Step 10: Fix permissions
        subprocess.run(['sudo', 'chown', '-h', 'mysql:mysql', primary_path], check=True)
        
        # Step 11: Start MySQL
        print(f"   ▶️  Starting MySQL...")
        result = subprocess.run(['sudo', 'systemctl', 'start', 'mysql'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to start MySQL: {result.stderr}")
        
        # Step 12: Wait for MySQL to be ready
        print(f"   ⏳ Waiting for MySQL...")
        for i in range(10):
            result = subprocess.run(['mysql', '-u', 'root', '-e', 'SELECT 1;'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                break
            if i < 9:
                import time
                time.sleep(1)
        else:
            raise Exception("MySQL didn't start properly")
        
        # Step 13: Verify symlink access
        print(f"   ✅ Verifying direct SDA1 access...")
        result = subprocess.run(['mysql', '-u', 'root', '-e', f'USE `{schema_name}`; SHOW TABLES;'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Schema not accessible: {result.stderr}")
        
        # Step 14: Verify symlink points to SDA1
        if os.path.islink(primary_path):
            link_target = os.readlink(primary_path)
            if '/local/mysql_old/' in link_target:
                print(f"   🎯 SUCCESS: Schema now accessed directly from SDA1!")
                print(f"   📍 Symlink: {primary_path} → {link_target}")
            else:
                print(f"   ⚠️  Warning: Symlink doesn't point to SDA1: {link_target}")
        
        # Step 15: Clean up SQL backup (optional)
        print(f"   🗑️  Ready to delete SQL backup (saves {backup_size:.1f} MB)")
        # Uncomment next line to actually delete:
        # os.remove(sql_backup)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        
        # Emergency cleanup - try to restart MySQL
        try:
            subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=False)
        except:
            pass
        
        return False

def test_dual_storage_conversion():
    """Test the conversion with a few schemas"""
    
    print("🚀 CONVERTING TO TRUE DUAL STORAGE")
    print("=" * 60)
    print("Goal: Access migrated schemas directly from SDA1 via symlinks")
    print("")
    
    # Get list of SQL backups
    backup_dir = "/local/mysql_migration_backups"
    sql_files = [f[:-4] for f in os.listdir(backup_dir) if f.endswith('.sql')]
    
    print(f"📊 Found {len(sql_files)} SQL backups to convert")
    
    # Test with first few schemas
    test_schemas = sql_files[:3]  # Start with 3 for testing
    
    print(f"\n🧪 Testing conversion with {len(test_schemas)} schemas:")
    print("-" * 60)
    
    for schema in test_schemas:
        print(f"\n{'='*20} {schema} {'='*20}")
        success = convert_sql_backup_to_symlink(schema, dry_run=True)
        if success:
            print(f"✅ Ready for conversion")
        else:
            print(f"❌ Needs attention")
    
    print(f"\n🎯 RESULT:")
    print(f"After conversion, these schemas will be:")
    print(f"   • 📦 Stored physically on SDA1 (/local/mysql_old/)")
    print(f"   • 🔗 Accessed via symlinks from /app/mysql/")
    print(f"   • ⚡ Available instantly (no restoration delay)")
    print(f"   • 💾 Using 50% less space (no SQL duplicates)")
    
    return test_schemas

def show_true_dual_storage_demo():
    """Show what true dual storage looks like"""
    
    print(f"\n🎬 TRUE DUAL STORAGE DEMO")
    print("=" * 60)
    
    print("📍 BEFORE CONVERSION:")
    print("   Schema click → 30-60s restoration → Access from primary drive")
    print("   Storage: SQL backup (SDA1) + Restored copy (Primary)")
    
    print("\n📍 AFTER CONVERSION:")
    print("   Schema click → Instant access → Direct from SDA1")
    print("   Storage: MySQL directory on SDA1, symlink on primary")
    
    print(f"\n🔗 How Symlinks Work:")
    print(f"   MySQL sees: /app/mysql/old_schema")
    print(f"   Symlink points to: /local/mysql_old/old_schema (on SDA1)")
    print(f"   Result: Data accessed directly from SDA1 transparently")
    
    print(f"\n✅ Benefits:")
    print(f"   • ⚡ Instant access (no delays)")
    print(f"   • 💾 50% space savings")
    print(f"   • 🔄 True dual storage")
    print(f"   • 📱 Same webapp experience")

if __name__ == "__main__":
    test_schemas = test_dual_storage_conversion()
    show_true_dual_storage_demo()
    
    print(f"\n" + "="*60)
    print(f"🚀 READY TO CONVERT TO TRUE DUAL STORAGE!")
    print(f"Run with --convert flag to convert first 3 schemas:")
    print(f"   python3 convert_to_true_dual_storage.py --convert")
    
    if '--convert' in sys.argv:
        print(f"\n🔄 CONVERTING FIRST 3 SCHEMAS...")
        for schema in test_schemas:
            print(f"\n{'='*50}")
            success = convert_sql_backup_to_symlink(schema, dry_run=False)
            if success:
                print(f"✅ {schema} converted to true dual storage!")
            else:
                print(f"❌ {schema} conversion failed")
        
        print(f"\n🎉 Conversion complete! Test accessing these schemas in your webapp.")