#!/usr/bin/env python3
"""
MySQL Setup Script for /dev/sda1
Sets up a secondary MySQL data directory on /dev/sda1 for old schemas.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

# Import database configuration
sys.path.append('/home/admin2/webapp_2')
from db_operations import DB_CONFIG

def run_command(cmd, shell=False, check=True, capture_output=False):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
            return result.stdout.strip() if result.stdout else ""
        else:
            subprocess.run(cmd, shell=shell, check=check)
            return ""
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        raise

def check_disk_space(path):
    """Check available disk space at given path."""
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        used_gb = usage.used / (1024**3)
        
        print(f"Disk space for {path}:")
        print(f"  Total: {total_gb:.2f} GB")
        print(f"  Used:  {used_gb:.2f} GB")
        print(f"  Free:  {free_gb:.2f} GB")
        
        return free_gb
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return 0

def check_mysql_status():
    """Check if MySQL is running and get current configuration."""
    try:
        # Check if MySQL service is running
        result = run_command(["systemctl", "is-active", "mysql"], capture_output=True, check=False)
        mysql_running = result == "active"
        
        if mysql_running:
            print("✅ MySQL service is running")
            
            # Get current datadir
            result = run_command(['mysql', '-u', 'root', '-e', 'SELECT @@datadir;'], 
                               capture_output=True, check=False)
            if result:
                current_datadir = result.split('\n')[1] if len(result.split('\n')) > 1 else "Unknown"
                print(f"📁 Current MySQL datadir: {current_datadir}")
            else:
                print("⚠️  Could not determine current MySQL datadir")
        else:
            print("❌ MySQL service is not running")
            
        return mysql_running
        
    except Exception as e:
        print(f"Error checking MySQL status: {e}")
        return False

def setup_sda1_directory():
    """Set up the MySQL data directory structure on /dev/sda1."""
    
    # Define paths - use the actual mount point /local
    sda1_mysql_path = Path("/local/mysql")
    sda1_mysql_data = sda1_mysql_path / "data"
    sda1_mysql_logs = sda1_mysql_path / "logs"
    sda1_mysql_tmp = sda1_mysql_path / "tmp"
    
    print("🔧 Setting up MySQL directory structure on /local (sda1 mount point)...")
    
    # Check if /local is mounted and accessible (this is where sda1 is mounted)
    if not Path("/local").exists():
        print("❌ /local does not exist or is not mounted")
        return False
    
    # Check available space (we need at least 100GB for the 83GB of data plus some buffer)
    free_space = check_disk_space("/local")
    if free_space < 100:
        print(f"⚠️  Warning: Only {free_space:.2f} GB free on /local (sda1), may not be sufficient")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Create directory structure
        for path in [sda1_mysql_path, sda1_mysql_data, sda1_mysql_logs, sda1_mysql_tmp]:
            path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {path}")
        
        # Set proper ownership and permissions
        mysql_user = "mysql"
        mysql_group = "mysql"
        
        # Change ownership to mysql:mysql
        run_command(["sudo", "chown", "-R", f"{mysql_user}:{mysql_group}", str(sda1_mysql_path)])
        
        # Set proper permissions
        run_command(["sudo", "chmod", "-R", "750", str(sda1_mysql_path)])
        
        print("✅ Directory structure created and permissions set")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up directory structure: {e}")
        return False

def create_mysql_config():
    """Create a MySQL configuration file for the secondary instance."""
    
    config_content = """
# MySQL Configuration for Secondary Instance on /local (sda1 mount point)
# This configuration allows MySQL to access data from both locations

[mysqld]
# Primary datadir (existing schemas on /dev/nvme0n1p3)
datadir = /var/lib/mysql

# Additional datadir for old schemas on /local (sda1 mount point)
# Note: MySQL doesn't natively support multiple datadirs
# We'll handle this through symbolic links or custom connection logic

# Security settings
bind-address = 127.0.0.1

# Performance settings
innodb_buffer_pool_size = 2G
innodb_log_file_size = 256M
innodb_flush_log_at_trx_commit = 2
innodb_flush_method = O_DIRECT

# Enable slow query log for monitoring
slow_query_log = 1
slow_query_log_file = /local/mysql/logs/mysql-slow.log
long_query_time = 2

# Error log
log_error = /local/mysql/logs/mysql-error.log

# Binary logging (optional, for replication/backup)
log_bin = /local/mysql/logs/mysql-bin
binlog_format = ROW
expire_logs_days = 7

# Temporary directory
tmpdir = /local/mysql/tmp

# Connection settings
max_connections = 200
connect_timeout = 60
wait_timeout = 28800
interactive_timeout = 28800

# MyISAM settings
key_buffer_size = 128M

# Query cache (if using older MySQL versions)
query_cache_type = 1
query_cache_size = 64M
"""
    
    config_file = "/local/mysql/my-sda1.cnf"
    
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Set proper ownership
        run_command(["sudo", "chown", "mysql:mysql", config_file])
        run_command(["sudo", "chmod", "644", config_file])
        
        print(f"✅ MySQL configuration created: {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating MySQL configuration: {e}")
        return False

def test_mysql_connection():
    """Test MySQL connection and basic functionality."""
    try:
        print("🔍 Testing MySQL connection...")
        
        # Test basic connection
        result = run_command(['mysql', '-u', 'root', '-e', 'SELECT VERSION();'], 
                           capture_output=True)
        if result:
            print(f"✅ MySQL connection successful: {result}")
        
        # Test showing databases
        result = run_command(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'], 
                           capture_output=True)
        if result:
            db_count = len([line for line in result.split('\n') if line.strip() and line != 'Database'])
            print(f"✅ Found {db_count} databases")
        
        return True
        
    except Exception as e:
        print(f"❌ MySQL connection test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 80)
    print("MySQL /dev/sda1 Setup for Old Schema Migration")
    print("=" * 80)
    
    # Check current system status
    print("\n📋 SYSTEM STATUS CHECK")
    print("-" * 40)
    
    # Check MySQL status
    mysql_running = check_mysql_status()
    if not mysql_running:
        print("❌ MySQL is not running. Please start MySQL first.")
        return 1
    
    # Check disk space on both locations
    print("\n💾 DISK SPACE CHECK")
    print("-" * 40)
    nvme_space = check_disk_space("/var/lib/mysql")
    sda1_space = check_disk_space("/dev/sda1")
    
    if sda1_space < 100:
        print("⚠️  Insufficient space on /dev/sda1 for migration")
    
    # Set up directory structure
    print("\n🔧 DIRECTORY SETUP")
    print("-" * 40)
    if not setup_sda1_directory():
        print("❌ Failed to set up directory structure")
        return 1
    
    # Create configuration
    print("\n⚙️  CONFIGURATION SETUP")
    print("-" * 40)
    if not create_mysql_config():
        print("❌ Failed to create MySQL configuration")
        return 1
    
    # Test connection
    print("\n🔍 CONNECTION TEST")
    print("-" * 40)
    if not test_mysql_connection():
        print("❌ MySQL connection test failed")
        return 1
    
    print("\n✅ SUCCESS!")
    print("=" * 80)
    print("📁 /local (sda1) MySQL setup completed successfully!")
    print(f"📊 Available space on /local: {sda1_space:.2f} GB")
    print("🔄 Ready for schema migration...")
    print("\nNext steps:")
    print("1. Run the migration script to move old schemas")
    print("2. Update the webapp to access schemas from both locations")
    print("3. Test dual-location access")
    
    return 0

if __name__ == "__main__":
    exit(main())