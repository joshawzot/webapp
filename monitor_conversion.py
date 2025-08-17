#!/usr/bin/env python3
"""
Monitor conversion progress without sudo prompts
"""

import os
import time
import subprocess

def count_converted_schemas():
    """Count schemas that have been converted to symlinks"""
    try:
        # Use ls without sudo by checking the MySQL data directory directly
        result = subprocess.run(['ls', '-la', '/app/mysql/'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.count('-> /local/mysql_old/')
        else:
            # Fallback: try to count by checking if directories exist in archive
            result = subprocess.run(['ls', '/local/mysql_old/'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return len([d for d in result.stdout.split('\n') if d.strip()])
    except:
        pass
    return 0

def count_total_backups():
    """Count total SQL backup files"""
    backup_dir = "/local/mysql_migration_backups"
    try:
        files = os.listdir(backup_dir)
        return len([f for f in files if f.endswith('.sql')])
    except:
        return 0

def monitor_progress():
    """Monitor conversion progress in real-time"""
    print("🔍 MONITORING CONVERSION PROGRESS")
    print("=" * 60)
    
    total_backups = count_total_backups()
    start_converted = count_converted_schemas()
    start_time = time.time()
    
    print(f"📊 Starting state:")
    print(f"   • Total SQL backups: {total_backups}")
    print(f"   • Already converted: {start_converted}")
    print(f"   • To convert: {total_backups}")
    print(f"\n🔄 Monitoring... (Ctrl+C to stop)")
    print("=" * 60)
    
    last_count = start_converted
    last_update = time.time()
    
    try:
        while True:
            current_converted = count_converted_schemas()
            current_time = time.time()
            
            if current_converted != last_count:
                # Progress made!
                elapsed = current_time - start_time
                rate = (current_converted - start_converted) / (elapsed / 3600) if elapsed > 0 else 0
                remaining = total_backups - current_converted
                eta_hours = remaining / rate if rate > 0 else 0
                
                print(f"⚡ Progress Update:")
                print(f"   ✅ Converted: {current_converted}/{total_backups} ({current_converted/total_backups*100:.1f}%)")
                print(f"   📈 Rate: {rate:.1f} schemas/hour")
                print(f"   ⏱️  ETA: {eta_hours:.1f} hours")
                print(f"   🕐 Elapsed: {elapsed/3600:.1f} hours")
                print("-" * 40)
                
                last_count = current_converted
                last_update = current_time
            
            # Check every 30 seconds
            time.sleep(30)
            
    except KeyboardInterrupt:
        final_converted = count_converted_schemas()
        total_elapsed = time.time() - start_time
        
        print(f"\n📊 FINAL STATUS:")
        print(f"   ✅ Total converted: {final_converted}")
        print(f"   ⏱️  Total time: {total_elapsed/3600:.1f} hours")
        print(f"   📈 Average rate: {(final_converted - start_converted)/(total_elapsed/3600):.1f} schemas/hour")

if __name__ == "__main__":
    monitor_progress()