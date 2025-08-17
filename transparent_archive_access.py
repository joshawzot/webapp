#!/usr/bin/env python3
"""
Transparent Archive Access Module

This module provides seamless access to archived schemas by automatically
restoring them temporarily when accessed, allowing the webapp to operate
on them as if they were on primary storage.
"""

import os
import subprocess
import mysql.connector
from datetime import datetime, timedelta
import json

class TransparentArchiveManager:
    """Manages transparent access to archived schemas"""
    
    def __init__(self):
        self.temp_restored_schemas = {}  # Track temporarily restored schemas
        self.temp_schema_file = '/tmp/temp_restored_schemas.json'
        self.load_temp_schema_cache()
    
    def load_temp_schema_cache(self):
        """Load list of temporarily restored schemas from cache"""
        try:
            if os.path.exists(self.temp_schema_file):
                with open(self.temp_schema_file, 'r') as f:
                    self.temp_restored_schemas = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load temp schema cache: {e}")
            self.temp_restored_schemas = {}
    
    def save_temp_schema_cache(self):
        """Save list of temporarily restored schemas to cache"""
        try:
            with open(self.temp_schema_file, 'w') as f:
                json.dump(self.temp_restored_schemas, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save temp schema cache: {e}")
    
    def is_schema_accessible(self, schema_name):
        """Check if schema is accessible in MySQL (either primary or temp restored)"""
        try:
            from db_operations import create_connection
            conn = create_connection()
            cursor = conn.cursor()
            cursor.execute("SHOW DATABASES LIKE %s", (schema_name,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result is not None
        except Exception:
            return False
    
    def is_schema_archived(self, schema_name):
        """Check if schema is in archive storage"""
        try:
            # Check if backup file exists (most reliable indicator)
            backup_file = self.get_backup_file_path(schema_name)
            if os.path.exists(backup_file):
                return True
            
            # Also check dual storage manager
            try:
                from dual_storage_db_operations import DualStorageManager
                dual_storage = DualStorageManager()
                location = dual_storage.get_schema_location(schema_name)
                return location == 'secondary' or 'migrated' in location
            except Exception:
                pass
            
            return False
        except Exception:
            return False
    
    def get_backup_file_path(self, schema_name):
        """Get path to backup file for schema"""
        return f"/local/mysql_migration_backups/{schema_name}.sql"
    
    def restore_schema_temporarily(self, schema_name):
        """Restore an archived schema temporarily for access"""
        try:
            # Check if backup file exists
            backup_file = self.get_backup_file_path(schema_name)
            if not os.path.exists(backup_file):
                raise Exception(f"Backup file not found: {backup_file}")
            
            # Check file size to ensure it's not empty
            file_size = os.path.getsize(backup_file)
            if file_size == 0:
                raise Exception(f"Backup file is empty: {backup_file}")
            
            print(f"🔄 Temporarily restoring archived schema: {schema_name}")
            print(f"   📁 Backup file: {backup_file} ({file_size/1024/1024:.1f} MB)")
            
            # Step 1: Create the database first
            print(f"   📝 Creating database {schema_name}...")
            create_db_result = subprocess.run(
                ['mysql', '-u', 'root', '-e', f'CREATE DATABASE IF NOT EXISTS `{schema_name}`;'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if create_db_result.returncode != 0:
                error_msg = create_db_result.stderr.strip() if create_db_result.stderr else "Unknown MySQL error"
                raise Exception(f"Database creation failed (code {create_db_result.returncode}): {error_msg}")
            
            # Step 2: Restore the tables into the database
            print(f"   📥 Restoring tables from backup...")
            result = subprocess.run(
                ['mysql', '-u', 'root', schema_name],
                stdin=open(backup_file, 'r'),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large schemas
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown MySQL error"
                # Try to clean up the partially created database
                subprocess.run(['mysql', '-u', 'root', '-e', f'DROP DATABASE IF EXISTS `{schema_name}`;'], capture_output=True)
                raise Exception(f"MySQL restore failed (code {result.returncode}): {error_msg}")
            
            # Verify the schema was created
            verify_result = subprocess.run(
                ['mysql', '-u', 'root', '-e', f'SHOW DATABASES LIKE "{schema_name}";'],
                capture_output=True, text=True
            )
            
            if verify_result.returncode != 0 or schema_name not in verify_result.stdout:
                raise Exception(f"Schema restoration verification failed - schema not found in MySQL")
            
            # Mark as temporarily restored
            self.temp_restored_schemas[schema_name] = {
                'restored_at': datetime.now(),
                'backup_file': backup_file,
                'originally_archived': True
            }
            self.save_temp_schema_cache()
            
            print(f"✅ Schema {schema_name} temporarily restored and verified")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restore schema {schema_name}: {e}")
            return False
    
    def ensure_schema_accessible(self, schema_name):
        """Ensure schema is accessible, restoring from archive if necessary"""
        
        # If schema is already accessible, we're good
        if self.is_schema_accessible(schema_name):
            return True
        
        # If schema is archived, restore it temporarily
        if self.is_schema_archived(schema_name):
            return self.restore_schema_temporarily(schema_name)
        
        # Schema not found anywhere
        return False
    
    def is_temp_restored(self, schema_name):
        """Check if schema was temporarily restored from archive"""
        return schema_name in self.temp_restored_schemas
    
    def get_schema_info(self, schema_name):
        """Get information about schema location and status"""
        info = {
            'name': schema_name,
            'accessible': self.is_schema_accessible(schema_name),
            'archived': self.is_schema_archived(schema_name),
            'temp_restored': self.is_temp_restored(schema_name),
            'status': 'unknown'
        }
        
        if info['accessible'] and info['temp_restored']:
            info['status'] = 'archived_temp_restored'
            info['storage_display'] = '📦 Archive (Temporary Access)'
            info['storage_color'] = 'warning'  # Orange/yellow for temp
        elif info['accessible'] and not info['archived']:
            info['status'] = 'primary'
            info['storage_display'] = '⚡ Primary Storage'
            info['storage_color'] = 'success'  # Green for primary
        elif info['archived'] and not info['accessible']:
            info['status'] = 'archived_inaccessible'
            info['storage_display'] = '📦 Archive (Not Accessible)'
            info['storage_color'] = 'secondary'  # Gray for inaccessible
        else:
            info['status'] = 'unknown'
            info['storage_display'] = '❓ Unknown Location'
            info['storage_color'] = 'danger'  # Red for unknown
        
        return info
    
    def cleanup_old_temp_restorations(self, max_age_hours=24):
        """Clean up old temporary restorations"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            schemas_to_remove = []
            
            for schema_name, info in self.temp_restored_schemas.items():
                if isinstance(info.get('restored_at'), str):
                    # Convert string back to datetime if needed
                    restored_at = datetime.fromisoformat(info['restored_at'].replace('Z', '+00:00'))
                else:
                    restored_at = info.get('restored_at', datetime.now())
                
                if restored_at < cutoff_time:
                    schemas_to_remove.append(schema_name)
            
            # Remove old temporary restorations
            cleaned_count = 0
            for schema_name in schemas_to_remove:
                try:
                    # Drop the database
                    from db_operations import create_connection
                    conn = create_connection()
                    cursor = conn.cursor()
                    cursor.execute(f"DROP DATABASE IF EXISTS `{schema_name}`")
                    cursor.close()
                    conn.close()
                    
                    # Remove from tracking
                    del self.temp_restored_schemas[schema_name]
                    cleaned_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not clean up {schema_name}: {e}")
            
            if cleaned_count > 0:
                self.save_temp_schema_cache()
                print(f"🧹 Cleaned up {cleaned_count} old temporary restorations")
            
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

# Global instance for the webapp to use
transparent_archive = TransparentArchiveManager()