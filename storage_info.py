#!/usr/bin/env python3
"""
MySQL Storage Information Utility

This script provides clear information about where MySQL data is actually stored,
resolving symbolic links and showing disk usage information.

Usage: python storage_info.py
"""

import os
import subprocess
import sys
import mysql.connector

def get_mysql_storage_info():
    """Get comprehensive information about MySQL data storage."""
    info = {}
    
    try:
        # Get MySQL's reported datadir
        print("🔍 Checking MySQL configuration...")
        result = subprocess.run(['mysql', '-u', 'root', '-e', 'SELECT @@datadir;'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            info['mysql_datadir'] = lines[1].strip() if len(lines) > 1 else "Unknown"
        else:
            info['mysql_datadir'] = "Could not connect to MySQL"
            
        # Check if MySQL datadir is a symbolic link
        mysql_path = '/var/lib/mysql'
        info['is_symlink'] = os.path.islink(mysql_path)
        
        if info['is_symlink']:
            info['symlink_target'] = os.readlink(mysql_path)
            info['resolved_path'] = os.path.realpath(mysql_path)
        else:
            info['symlink_target'] = None
            info['resolved_path'] = mysql_path
            
        # Get disk information for the actual storage location
        storage_path = info['resolved_path'] if info['is_symlink'] else '/var/lib/mysql'
        df_result = subprocess.run(['df', '-h', storage_path], capture_output=True, text=True)
        
        if df_result.returncode == 0:
            lines = df_result.stdout.strip().split('\n')
            if len(lines) > 1:
                disk_info = lines[1].split()
                info['disk_device'] = disk_info[0]
                info['disk_size'] = disk_info[1]
                info['disk_used'] = disk_info[2]
                info['disk_available'] = disk_info[3]
                info['disk_usage_percent'] = disk_info[4]
                info['mount_point'] = disk_info[5]
        
        # Get MySQL data directory size
        try:
            du_result = subprocess.run(['sudo', 'du', '-sh', storage_path], 
                                     capture_output=True, text=True)
            if du_result.returncode == 0:
                info['mysql_data_size'] = du_result.stdout.split()[0]
            else:
                # Try without sudo
                du_result = subprocess.run(['du', '-sh', storage_path], 
                                         capture_output=True, text=True)
                info['mysql_data_size'] = du_result.stdout.split()[0] if du_result.returncode == 0 else "Permission denied"
        except:
            info['mysql_data_size'] = "Could not calculate"
            
        # Get database count
        try:
            db_result = subprocess.run(['mysql', '-u', 'root', '-e', 
                                      'SELECT COUNT(*) FROM information_schema.SCHEMATA;'], 
                                     capture_output=True, text=True)
            if db_result.returncode == 0:
                lines = db_result.stdout.strip().split('\n')
                info['database_count'] = lines[1].strip() if len(lines) > 1 else "Unknown"
        except:
            info['database_count'] = "Could not count"
            
    except Exception as e:
        info['error'] = str(e)
        
    return info

def print_storage_report(info):
    """Print a formatted report of MySQL storage information."""
    print("\n" + "="*60)
    print("           MySQL STORAGE INFORMATION REPORT")
    print("="*60)
    
    if 'error' in info:
        print(f"❌ Error: {info['error']}")
        return
        
    print(f"\n📊 MYSQL CONFIGURATION:")
    print(f"   Reported datadir: {info.get('mysql_datadir', 'Unknown')}")
    print(f"   Database count:   {info.get('database_count', 'Unknown')}")
    
    print(f"\n🔗 SYMBOLIC LINK STATUS:")
    if info.get('is_symlink'):
        print(f"   ✅ /var/lib/mysql IS a symbolic link")
        print(f"   🎯 Points to: {info.get('symlink_target', 'Unknown')}")
        print(f"   📁 Resolved path: {info.get('resolved_path', 'Unknown')}")
        print(f"   ⚠️  Data is NOT on the root filesystem!")
    else:
        print(f"   ❌ /var/lib/mysql is NOT a symbolic link")
        print(f"   📁 Data stored directly at: {info.get('resolved_path', 'Unknown')}")
        
    print(f"\n💾 PHYSICAL STORAGE:")
    print(f"   Disk device:      {info.get('disk_device', 'Unknown')}")
    print(f"   Mount point:      {info.get('mount_point', 'Unknown')}")
    print(f"   Total disk size:  {info.get('disk_size', 'Unknown')}")
    print(f"   Used space:       {info.get('disk_used', 'Unknown')}")
    print(f"   Available space:  {info.get('disk_available', 'Unknown')}")
    print(f"   Usage percentage: {info.get('disk_usage_percent', 'Unknown')}")
    
    print(f"\n📈 MYSQL DATA SIZE:")
    print(f"   MySQL data size:  {info.get('mysql_data_size', 'Unknown')}")
    
    print(f"\n✅ VERIFICATION COMMANDS:")
    print(f"   ls -l /var/lib/mysql")
    print(f"   readlink -f /var/lib/mysql")
    print(f"   df -h {info.get('mount_point', '/app')}")
    print(f"   du -sh {info.get('resolved_path', '/app/mysql')}")
    
    print("\n" + "="*60)
    
    # Warning if data is redirected
    if info.get('is_symlink'):
        print("⚠️  WARNING: MySQL data location is redirected!")
        print("   This means the data is NOT where MySQL thinks it is.")
        print("   Always check for symbolic links in production systems.")
        print("="*60)

def main():
    """Main function to run the storage information utility."""
    print("MySQL Storage Information Utility")
    print("Checking actual storage location for MySQL data...")
    
    info = get_mysql_storage_info()
    print_storage_report(info)
    
    # Check if we can connect to MySQL
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        connection.close()
        print("\n✅ MySQL connection test: SUCCESS")
    except Exception as e:
        print(f"\n❌ MySQL connection test: FAILED - {e}")

if __name__ == "__main__":
    main()