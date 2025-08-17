#!/usr/bin/env python3
"""
Quick script to check conversion progress
"""

import subprocess
import os

def check_progress():
    print("🔍 CHECKING CONVERSION PROGRESS")
    print("=" * 50)
    
    # Count total SQL backups
    backup_dir = "/local/mysql_migration_backups"
    try:
        sql_files = [f for f in os.listdir(backup_dir) if f.endswith('.sql')]
        total_backups = len(sql_files)
    except:
        total_backups = 0
    
    # Count symlinks (converted schemas)
    converted_count = 0
    try:
        result = subprocess.run(['sudo', 'ls', '-la', '/app/mysql/'], 
                              capture_output=True, text=True)
        converted_count = result.stdout.count('-> /local/mysql_old/')
    except:
        pass
    
    # Calculate progress
    remaining = total_backups
    progress_pct = (converted_count / (converted_count + remaining)) * 100 if (converted_count + remaining) > 0 else 0
    
    print(f"📊 Progress: {converted_count}/{converted_count + remaining} ({progress_pct:.1f}%)")
    print(f"   ✅ Direct SDA1 access: {converted_count}")
    print(f"   🔄 Still converting: {remaining}")
    
    if remaining > 0:
        estimated_hours = remaining * 1.5 / 60
        print(f"   ⏱️  Est. time remaining: {estimated_hours:.1f} hours")
    else:
        print(f"   🎉 ALL CONVERSIONS COMPLETE!")
    
    print(f"\n💡 Run this script again to check progress:")
    print(f"   python3 check_conversion_progress.py")

if __name__ == "__main__":
    check_progress()