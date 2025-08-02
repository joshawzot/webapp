#!/usr/bin/env python3
"""
Quick MySQL Storage Check

A simple script to quickly verify where MySQL data is actually stored.
Run this when you need to quickly check the storage location.
"""

import os
import subprocess

def quick_storage_check():
    """Perform a quick check of MySQL storage location."""
    print("🔍 Quick MySQL Storage Check")
    print("-" * 40)
    
    # Check symbolic link
    if os.path.islink('/var/lib/mysql'):
        target = os.readlink('/var/lib/mysql')
        print(f"✅ /var/lib/mysql -> {target}")
    else:
        print("❌ /var/lib/mysql is NOT a symbolic link")
    
    # Check disk usage
    try:
        result = subprocess.run(['df', '-h', '/app'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                disk_info = lines[1].split()
                print(f"💾 Disk: {disk_info[0]} ({disk_info[1]} total, {disk_info[2]} used)")
    except:
        print("❌ Could not get disk information")
    
    # Check MySQL status
    try:
        result = subprocess.run(['mysql', '-u', 'root', '-e', 'SELECT COUNT(*) FROM information_schema.SCHEMATA;'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"📊 Databases: {lines[1]} total")
    except:
        print("❌ Could not connect to MySQL")

if __name__ == "__main__":
    quick_storage_check()