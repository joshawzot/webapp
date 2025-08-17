#!/usr/bin/env python3
"""
Symlink-aware storage detection for proper dual storage
"""

import os
import subprocess
from datetime import datetime, timedelta

class SymlinkAwareStorage:
    """Storage manager that properly detects symlinked archived schemas"""
    
    def __init__(self):
        self.primary_storage = '/app/mysql'
        self.archive_storage = '/local/mysql_old'
        self.backup_storage = '/local/mysql_migration_backups'
    
    def get_schema_storage_info(self, schema_name):
        """Get comprehensive storage information for a schema"""
        
        primary_path = os.path.join(self.primary_storage, schema_name)
        archive_path = os.path.join(self.archive_storage, schema_name)
        backup_path = os.path.join(self.backup_storage, f"{schema_name}.sql")
        
        info = {
            'name': schema_name,
            'location': 'unknown',
            'storage_device': 'Unknown',
            'storage_path': 'Unknown', 
            'storage_type': 'Unknown',
            'storage_color': 'secondary',
            'access_method': 'unknown',
            'is_symlink': False,
            'archive_exists': False,
            'backup_exists': False
        }
        
        # Check various storage states
        try:
            # Check if backup file exists
            info['backup_exists'] = os.path.exists(backup_path)
            
            # Check if archive directory exists
            info['archive_exists'] = os.path.exists(archive_path)
            
            # Check primary storage
            if os.path.exists(primary_path):
                if os.path.islink(primary_path):
                    # It's a symlink - proper archive storage
                    link_target = os.readlink(primary_path)
                    info['is_symlink'] = True
                    info['location'] = 'archived_symlinked'
                    info['storage_device'] = '/dev/sda1 (Archive - Direct Access)'
                    info['storage_path'] = link_target
                    info['storage_type'] = '📦 Archive Storage (Direct Access)'
                    info['storage_color'] = 'info'
                    info['access_method'] = 'direct_symlink'
                    
                elif os.path.isdir(primary_path):
                    # Real directory in primary storage
                    info['location'] = 'primary' 
                    info['storage_device'] = '/dev/nvme2n1p1 (Primary)'
                    info['storage_path'] = primary_path
                    info['storage_type'] = '⚡ Primary Storage'
                    info['storage_color'] = 'success'
                    info['access_method'] = 'direct_primary'
                    
            elif info['archive_exists']:
                # In archive but not symlinked
                info['location'] = 'archived_not_linked'
                info['storage_device'] = '/dev/sda1 (Archive - Not Accessible)'
                info['storage_path'] = archive_path
                info['storage_type'] = '📦 Archive Storage (Not Linked)'
                info['storage_color'] = 'warning'
                info['access_method'] = 'needs_linking'
                
            elif info['backup_exists']:
                # Only SQL backup exists
                info['location'] = 'backup_only'
                info['storage_device'] = '/dev/sda1 (SQL Backup)'
                info['storage_path'] = backup_path
                info['storage_type'] = '💾 SQL Backup (Needs Restore)'
                info['storage_color'] = 'secondary'
                info['access_method'] = 'needs_restoration'
                
            else:
                # Not found anywhere
                info['location'] = 'not_found'
                info['storage_type'] = '❓ Not Found'
                info['storage_color'] = 'danger'
                info['access_method'] = 'unavailable'
                
        except Exception as e:
            print(f"Error checking storage for {schema_name}: {e}")
            
        return info
    
    def is_schema_accessible(self, schema_name):
        """Check if schema is accessible to MySQL"""
        try:
            result = subprocess.run(
                ['mysql', '-u', 'root', '-e', f'SHOW DATABASES LIKE "{schema_name}";'],
                capture_output=True, text=True
            )
            return result.returncode == 0 and schema_name in result.stdout
        except Exception:
            return False
    
    def get_accessible_schemas_by_type(self):
        """Get schemas categorized by their storage type"""
        
        categories = {
            'primary': [],
            'archived_symlinked': [],
            'archived_not_linked': [],
            'backup_only': [],
            'not_found': []
        }
        
        # Get all databases from MySQL
        try:
            result = subprocess.run(
                ['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                databases = [line.strip() for line in result.stdout.split('\n')[1:] if line.strip()]
                
                # Filter out system databases
                user_databases = [db for db in databases if db not in 
                                ['information_schema', 'performance_schema', 'mysql', 'sys', 'Database']]
                
                for db in user_databases:
                    info = self.get_schema_storage_info(db)
                    categories[info['location']].append({
                        'name': db,
                        'info': info
                    })
        
        except Exception as e:
            print(f"Error categorizing schemas: {e}")
            
        return categories
    
    def create_symlink_for_archived_schema(self, schema_name):
        """Create symlink for a schema that exists in archive but isn't linked"""
        
        primary_path = os.path.join(self.primary_storage, schema_name)
        archive_path = os.path.join(self.archive_storage, schema_name)
        
        if not os.path.exists(archive_path):
            raise Exception(f"Archive directory doesn't exist: {archive_path}")
            
        if os.path.exists(primary_path):
            raise Exception(f"Primary path already exists: {primary_path}")
        
        try:
            print(f"🔗 Creating symlink for {schema_name}...")
            
            # Stop MySQL temporarily
            subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], check=True)
            
            # Create symlink
            subprocess.run(['sudo', 'ln', '-s', archive_path, primary_path], check=True)
            
            # Fix permissions
            subprocess.run(['sudo', 'chown', '-h', 'mysql:mysql', primary_path], check=True)
            
            # Start MySQL
            subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=True)
            
            print(f"✅ Symlink created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create symlink: {e}")
            # Try to restart MySQL
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=False)
            except:
                pass
            return False

# Global instance
symlink_storage = SymlinkAwareStorage()