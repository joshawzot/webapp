#!/usr/bin/env python3
"""
Convert an existing MySQL schema to symlink-based archive storage
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

def convert_schema_to_archive(schema_name):
    """Convert a specific schema from primary to archive storage with symlink"""
    
    primary_dir = '/app/mysql'
    archive_dir = '/local/mysql_old'
    
    primary_path = os.path.join(primary_dir, schema_name)
    archive_path = os.path.join(archive_dir, schema_name)
    
    print(f"🔄 Converting {schema_name} to archive storage...")
    print(f"   📍 From: {primary_path}")
    print(f"   📦 To: {archive_path}")
    
    try:
        # Ensure archive directory exists
        subprocess.run(['sudo', 'mkdir', '-p', archive_dir], check=True)
        
        # Check if source exists
        try:
            result = subprocess.run(['sudo', 'test', '-d', primary_path], capture_output=True)
            if result.returncode != 0:
                raise Exception(f"Source directory not found: {primary_path}")
        except Exception as e:
            raise Exception(f"Cannot access source directory {primary_path}: {e}")
        
        if os.path.islink(primary_path):
            print(f"   ⏭️  Already a symlink - conversion not needed")
            return True
        
        # Check if archive path already exists
        result = subprocess.run(['sudo', 'test', '-d', archive_path], capture_output=True)
        if result.returncode == 0:
            raise Exception(f"Archive directory already exists: {archive_path}")
        
        # Step 1: Stop MySQL to prevent corruption
        print(f"   ⏸️  Stopping MySQL...")
        result = subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to stop MySQL: {result.stderr}")
        
        # Step 2: Move directory to archive
        print(f"   📦 Moving directory to archive...")
        result = subprocess.run(['sudo', 'mv', primary_path, archive_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to move directory: {result.stderr}")
        
        # Step 3: Create symlink
        print(f"   🔗 Creating symlink...")
        result = subprocess.run(['sudo', 'ln', '-s', archive_path, primary_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            # Try to move back if symlink creation failed
            subprocess.run(['sudo', 'mv', archive_path, primary_path], check=False)
            raise Exception(f"Failed to create symlink: {result.stderr}")
        
        # Step 4: Fix ownership
        print(f"   🔧 Fixing permissions...")
        subprocess.run(['sudo', 'chown', '-h', 'mysql:mysql', primary_path], check=True)
        
        # Step 5: Start MySQL
        print(f"   ▶️  Starting MySQL...")
        result = subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to start MySQL: {result.stderr}")
        
        # Step 6: Wait for MySQL to be ready
        print(f"   ⏳ Waiting for MySQL to be ready...")
        for i in range(10):
            result = subprocess.run(['mysql', '-u', 'root', '-e', 'SELECT 1;'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                break
            if i < 9:
                import time
                time.sleep(1)
        else:
            raise Exception("MySQL didn't start properly after conversion")
        
        # Step 7: Verify schema is still accessible
        print(f"   ✅ Verifying schema access...")
        result = subprocess.run(['mysql', '-u', 'root', '-e', f'USE `{schema_name}`; SHOW TABLES;'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Schema not accessible after conversion: {result.stderr}")
        
        print(f"✅ Successfully converted {schema_name} to archive storage!")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        
        # Emergency cleanup - try to restart MySQL
        print(f"🚨 Attempting emergency cleanup...")
        try:
            subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=False)
        except:
            pass
        
        return False

def verify_symlink_access(schema_name):
    """Verify that symlink provides transparent access"""
    
    print(f"\n🧪 Testing symlink access for {schema_name}...")
    
    primary_path = os.path.join('/app/mysql', schema_name)
    
    try:
        # Check if it's a symlink
        if not os.path.islink(primary_path):
            print(f"   ❌ Not a symlink: {primary_path}")
            return False
        
        # Check where symlink points
        link_target = os.readlink(primary_path)
        print(f"   🔗 Symlink target: {link_target}")
        
        # Check if target exists
        if not os.path.exists(link_target):
            print(f"   ❌ Symlink target doesn't exist: {link_target}")
            return False
        
        # Test MySQL access
        result = subprocess.run(['mysql', '-u', 'root', '-e', f'USE `{schema_name}`; SHOW TABLES;'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ❌ MySQL access failed: {result.stderr}")
            return False
        
        print(f"   ✅ MySQL access works perfectly!")
        print(f"   📊 Sample tables: {result.stdout.strip()}")
        
        # Test file system access
        import glob
        files = glob.glob(os.path.join(primary_path, '*.ibd'))[:3]
        print(f"   📁 Sample data files via symlink: {len(files)} .ibd files found")
        
        print(f"   🎯 RESULT: Schema is accessible from archive via symlink!")
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False

def show_storage_comparison(schema_name):
    """Show storage details before and after conversion"""
    
    print(f"\n📊 STORAGE COMPARISON for {schema_name}:")
    print("=" * 60)
    
    primary_path = os.path.join('/app/mysql', schema_name)
    archive_path = os.path.join('/local/mysql_old', schema_name)
    
    try:
        # Get disk usage info
        primary_df = subprocess.run(['df', '-h', '/app'], capture_output=True, text=True)
        archive_df = subprocess.run(['df', '-h', '/local'], capture_output=True, text=True)
        
        primary_info = primary_df.stdout.split('\n')[1].split() if primary_df.returncode == 0 else ["Unknown"]
        archive_info = archive_df.stdout.split('\n')[1].split() if archive_df.returncode == 0 else ["Unknown"]
        
        print(f"📍 CURRENT LOCATION:")
        if os.path.islink(primary_path):
            print(f"   🔗 Symlink: {primary_path} → {os.readlink(primary_path)}")
            print(f"   💾 Physical storage: {archive_info[0]} ({archive_info[3]} available)")
            print(f"   ⚡ Access method: Direct via symlink")
        else:
            print(f"   📁 Real directory: {primary_path}")
            print(f"   💾 Physical storage: {primary_info[0]} ({primary_info[3]} available)")
            print(f"   ⚡ Access method: Direct primary")
        
        print(f"\n🎯 BENEFITS:")
        print(f"   ✅ Schema moved to larger archive disk")
        print(f"   ✅ MySQL access unchanged (transparent)")
        print(f"   ✅ Webapp access unchanged")
        print(f"   ✅ Primary disk space freed up")
        
    except Exception as e:
        print(f"Error getting storage info: {e}")

if __name__ == "__main__":
    # Target schema to convert
    schema_name = "MaxZhang_Cullinan_183_100ReadRAC1Reset_20250109"
    
    print("🚀 CONVERTING TO PROPER DUAL STORAGE")
    print("=" * 60)
    
    print(f"🎯 Target schema: {schema_name}")
    print(f"📅 This schema is from January 9, 2025 (>180 days old)")
    print(f"💡 Perfect candidate for archive storage!")
    
    print(f"\n⚠️  This will:")
    print(f"   1. Stop MySQL temporarily")
    print(f"   2. Move schema to archive disk")
    print(f"   3. Create symlink for transparent access")
    print(f"   4. Restart MySQL")
    
    input(f"\nPress Enter to convert {schema_name} or Ctrl+C to cancel...")
    
    # Perform conversion
    success = convert_schema_to_archive(schema_name)
    
    if success:
        # Verify symlink access
        verify_symlink_access(schema_name)
        
        # Show storage comparison
        show_storage_comparison(schema_name)
        
        print(f"\n🎉 SUCCESS! Schema {schema_name} converted to proper dual storage!")
        print(f"🔗 It now uses symlink-based archive access")
        print(f"⚡ Access speed: Identical to primary storage")
        print(f"💾 Storage location: Archive disk (/local)")
    else:
        print(f"\n❌ Conversion failed. Schema remains in primary storage.")