#!/usr/bin/env python3
"""
Proper Dual Storage Implementation

This implements true dual storage by moving actual MySQL data directories
to archive storage and using symbolic links for direct access.
"""

import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta

class ProperDualStorage:
    """Proper dual storage using MySQL data directory movement and symlinks"""
    
    def __init__(self):
        self.primary_storage = '/app/mysql'
        self.archive_storage = '/local/mysql_old'
        self.cutoff_days = 60
        
        # Ensure archive directory exists
        os.makedirs(self.archive_storage, exist_ok=True)
    
    def extract_timestamp_from_schema_name(self, schema_name):
        """Extract timestamp from schema name if it exists."""
        import re
        patterns = [
            r'_(\d{14})$',  # _20250801231213 at end
            r'_(\d{12})$',  # _202508012312 at end (12 digits)
            r'_(\d{8})$',   # _20250801 at end (8 digits - date only)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, schema_name)
            if match:
                timestamp_str = match.group(1)
                try:
                    if len(timestamp_str) == 14:  # YYYYMMDDHHMMSS
                        return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                    elif len(timestamp_str) == 12:  # YYYYMMDDHHMM
                        return datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                    elif len(timestamp_str) == 8:   # YYYYMMDD
                        return datetime.strptime(timestamp_str, '%Y%m%d')
                except ValueError:
                    continue
        return None
    
    def should_be_archived(self, schema_name):
        """Check if schema should be in archive storage based on age"""
        timestamp = self.extract_timestamp_from_schema_name(schema_name)
        if timestamp is None:
            return False  # No timestamp, keep in primary
        
        cutoff_date = datetime.now() - timedelta(days=self.cutoff_days)
        return timestamp < cutoff_date
    
    def get_schema_location(self, schema_name):
        """Get the actual current location of a schema"""
        primary_path = os.path.join(self.primary_storage, schema_name)
        archive_path = os.path.join(self.archive_storage, schema_name)
        
        # Check if it's a symlink in primary pointing to archive
        if os.path.islink(primary_path):
            if os.path.exists(archive_path):
                return 'archived_linked'  # In archive, linked from primary
            else:
                return 'broken_link'  # Broken symlink
        
        # Check if it's a real directory in primary
        elif os.path.isdir(primary_path):
            return 'primary'  # Actually in primary storage
        
        # Check if it's only in archive (no link)
        elif os.path.isdir(archive_path):
            return 'archived_only'  # In archive but not linked
        
        else:
            return 'not_found'  # Doesn't exist anywhere
    
    def migrate_schema_to_archive(self, schema_name):
        """Move a schema from primary to archive storage with symlink"""
        try:
            primary_path = os.path.join(self.primary_storage, schema_name)
            archive_path = os.path.join(self.archive_storage, schema_name)
            
            if not os.path.isdir(primary_path):
                raise Exception(f"Schema directory not found in primary: {primary_path}")
            
            if os.path.islink(primary_path):
                raise Exception(f"Schema is already a symlink: {schema_name}")
            
            if os.path.exists(archive_path):
                raise Exception(f"Schema already exists in archive: {archive_path}")
            
            print(f"🔄 Moving {schema_name} to archive storage...")
            
            # Step 1: Stop MySQL temporarily to avoid corruption
            print("   ⏸️  Stopping MySQL...")
            subprocess.run(['sudo', 'systemctl', 'stop', 'mysql'], check=True)
            
            # Step 2: Move the directory to archive
            print("   📦 Moving directory to archive...")
            subprocess.run(['sudo', 'mv', primary_path, archive_path], check=True)
            
            # Step 3: Create symlink from primary to archive
            print("   🔗 Creating symlink...")
            subprocess.run(['sudo', 'ln', '-s', archive_path, primary_path], check=True)
            
            # Step 4: Fix ownership and permissions
            print("   🔧 Fixing permissions...")
            subprocess.run(['sudo', 'chown', '-h', 'mysql:mysql', primary_path], check=True)
            
            # Step 5: Start MySQL
            print("   ▶️  Starting MySQL...")
            subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=True)
            
            # Step 6: Verify MySQL started correctly
            print("   ✅ Verifying MySQL...")
            result = subprocess.run(['sudo', 'systemctl', 'is-active', 'mysql'], 
                                  capture_output=True, text=True)
            if result.stdout.strip() != 'active':
                raise Exception("MySQL failed to start after migration")
            
            print(f"✅ Successfully migrated {schema_name} to archive")
            return True
            
        except Exception as e:
            print(f"❌ Failed to migrate {schema_name}: {e}")
            # Try to restart MySQL if it's stopped
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'mysql'], check=False)
            except:
                pass
            return False
    
    def list_schemas_needing_migration(self):
        """List schemas that should be migrated to archive"""
        try:
            # Get all schemas from primary storage
            primary_schemas = []
            if os.path.exists(self.primary_storage):
                for item in os.listdir(self.primary_storage):
                    item_path = os.path.join(self.primary_storage, item)
                    # Skip if it's already a symlink (already migrated)
                    if os.path.isdir(item_path) and not os.path.islink(item_path):
                        if self.should_be_archived(item):
                            primary_schemas.append(item)
            
            return primary_schemas
            
        except Exception as e:
            print(f"Error listing schemas: {e}")
            return []
    
    def get_storage_info(self, schema_name):
        """Get comprehensive storage information for a schema"""
        location = self.get_schema_location(schema_name)
        timestamp = self.extract_timestamp_from_schema_name(schema_name)
        should_archive = self.should_be_archived(schema_name)
        
        info = {
            'name': schema_name,
            'location': location,
            'timestamp': timestamp,
            'should_be_archived': should_archive,
            'age_days': None,
            'storage_device': 'Unknown',
            'storage_path': 'Unknown',
            'storage_type': 'Unknown',
            'storage_color': 'secondary'
        }
        
        if timestamp:
            info['age_days'] = (datetime.now() - timestamp).days
        
        if location == 'primary':
            info['storage_device'] = '/dev/nvme2n1p1 (Primary)'
            info['storage_path'] = self.primary_storage
            info['storage_type'] = '⚡ Primary Storage'
            info['storage_color'] = 'success'
        elif location == 'archived_linked':
            info['storage_device'] = '/dev/sda1 (Archive)'
            info['storage_path'] = self.archive_storage
            info['storage_type'] = '📦 Archive Storage (Direct Access)'
            info['storage_color'] = 'info'
        elif location == 'archived_only':
            info['storage_device'] = '/dev/sda1 (Archive)'
            info['storage_path'] = self.archive_storage
            info['storage_type'] = '📦 Archive Storage (Not Linked)'
            info['storage_color'] = 'warning'
        elif location == 'broken_link':
            info['storage_type'] = '❌ Broken Link'
            info['storage_color'] = 'danger'
        
        return info

# Global instance
proper_dual_storage = ProperDualStorage()