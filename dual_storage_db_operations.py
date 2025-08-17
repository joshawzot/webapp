#!/usr/bin/env python3
"""
Dual Storage Database Operations
Enhanced database operations to handle schemas stored in both /dev/nvme0n1p3 and /dev/sda1
"""

import mysql.connector
from urllib.parse import quote_plus
import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

# Import original db_operations for compatibility
sys.path.append('/home/admin2/webapp_2')
from db_operations import *  # Import all original functions
from analyze_schema_ages import extract_timestamp_from_name

class DualStorageManager:
    """Manages database access across dual storage locations."""
    
    def __init__(self):
        self.primary_storage = "/var/lib/mysql"  # /dev/nvme0n1p3
        self.secondary_storage = "/local/mysql/data"  # /dev/sda1 mounted at /local
        self.storage_map_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.last_cache_update = 0  # Force fresh cache on next update
        
    def get_storage_info(self):
        """Get information about both storage locations using MySQL queries."""
        import subprocess
        
        storage_info = {
            'primary': {
                'path': self.primary_storage,
                'device': '/dev/nvme0n1p3',
                'schemas': [],
                'total_size': 0,
                'free_space': 0,
                'schema_count': 0
            },
            'secondary': {
                'path': self.secondary_storage,
                'device': '/local',
                'schemas': [],
                'total_size': 0,
                'free_space': 0,
                'schema_count': 0
            }
        }
        
        try:
            # Get MySQL datadir information
            conn = create_connection()
            cursor = conn.cursor()
            
            # Get MySQL's actual datadir
            cursor.execute("SELECT @@datadir")
            mysql_datadir = cursor.fetchone()[0]
            storage_info['primary']['path'] = mysql_datadir
            
            # Get database count using MySQL
            cursor.execute("SHOW DATABASES")
            all_databases = cursor.fetchall()
            user_databases = [db[0] for db in all_databases 
                            if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
            
            # For now, assume all schemas are on primary storage
            # This will be updated after migration when we have actual symbolic links
            storage_info['primary']['schema_count'] = len(user_databases)
            storage_info['secondary']['schema_count'] = 0
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error getting MySQL storage info: {e}")
        
        # Get disk usage information using safer methods
        for location in ['primary', 'secondary']:
            device = storage_info[location]['device']
            try:
                # Try to get disk usage using df command instead of direct file access
                result = subprocess.run(['df', '-B1', device], 
                                      capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        fields = lines[1].split()
                        if len(fields) >= 4:
                            total_space = int(fields[1])
                            free_space = int(fields[3])
                            storage_info[location]['total_space'] = total_space
                            storage_info[location]['free_space'] = free_space
                            
            except Exception as e:
                print(f"Could not get disk usage for {device}: {e}")
                # Set reasonable defaults
                storage_info[location]['total_space'] = 0
                storage_info[location]['free_space'] = 0
        
        return storage_info
    
    def update_storage_map(self):
        """Update the cache of which schemas are stored where by checking actual locations."""
        current_time = datetime.now().timestamp()
        
        # Check if cache is still valid
        if (current_time - self.last_cache_update) < self.cache_timeout:
            return
        
        print("🔄 Updating storage location cache...")
        storage_map = {}
        
        try:
            # Get all databases from MySQL
            conn = create_connection()
            cursor = conn.cursor()
            
            cursor.execute("SHOW DATABASES")
            all_databases = cursor.fetchall()
            user_databases = [db[0] for db in all_databases 
                            if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
            
            cursor.close()
            conn.close()
            
            # Check actual storage locations (post-nuclear-fix reality)
            for schema_name in user_databases:
                # Method 1: Check if schema ACTUALLY exists in archive location
                # (After nuclear fix, most should be moved back to primary)
                try:
                    import subprocess
                    result = subprocess.run(
                        ['sudo', 'ls', '-d', f'{self.secondary_storage}/{schema_name}'],
                        capture_output=True, text=True, check=False
                    )
                    if result.returncode == 0:
                        storage_map[schema_name] = 'secondary'
                        continue
                except Exception as e:
                    # Fall through to primary
                    pass
                
                # Method 2: Check if schema exists in primary location  
                # (This is now the default after nuclear fix)
                try:
                    import subprocess
                    result = subprocess.run(
                        ['sudo', 'ls', '-d', f'{self.primary_storage}/{schema_name}'],
                        capture_output=True, text=True, check=False
                    )
                    if result.returncode == 0:
                        storage_map[schema_name] = 'primary'
                        continue
                except Exception as e:
                    pass
                
                # Method 3: Fallback - if visible to MySQL but can't find files, assume primary
                # (This handles any edge cases or permission issues)
                storage_map[schema_name] = 'primary'
            
            self.storage_map_cache = storage_map
            self.last_cache_update = current_time
            
            # Count schemas by location for reporting
            location_counts = {}
            for location in storage_map.values():
                location_counts[location] = location_counts.get(location, 0) + 1
            
            print(f"✅ Storage map updated: {len(storage_map)} schemas tracked")
            print(f"   Primary: {location_counts.get('primary', 0)} schemas")
            print(f"   Archive: {location_counts.get('secondary', 0)} schemas")
            
        except Exception as e:
            print(f"Error updating storage map: {e}")
            # Keep existing cache if update fails
            self.last_cache_update = current_time
    
    def get_schema_location(self, schema_name):
        """Get the storage location of a specific schema."""
        # Fast path: Check cache first
        if schema_name in self.storage_map_cache:
            return self.storage_map_cache.get(schema_name, 'primary')
        
        # Check if schema is visible to MySQL (determines if it's accessible)
        try:
            import subprocess
            result = subprocess.run(
                ['mysql', '-u', 'root', '-e', f'SHOW DATABASES LIKE "{schema_name}";'],
                capture_output=True, text=True, check=False
            )
            
            if result.returncode == 0 and schema_name in result.stdout:
                # Schema is visible to MySQL, it's on primary storage
                self.storage_map_cache[schema_name] = 'primary'
                return 'primary'
            else:
                # Schema not visible to MySQL, might be in archive
                # Quick check for migration log to see if it was migrated
                migration_log_path = Path('/home/admin2/webapp_2').glob('safe_migration_log_*.json')
                migration_logs = list(migration_log_path)
                
                if migration_logs:
                    # Check most recent migration log
                    latest_log = sorted(migration_logs)[-1]
                    try:
                        import json
                        with open(latest_log, 'r') as f:
                            migration_data = json.load(f)
                        
                        # Check if this schema was successfully migrated
                        for entry in migration_data:
                            if (entry.get('schema') == schema_name and 
                                entry.get('status') == 'success'):
                                self.storage_map_cache[schema_name] = 'secondary'
                                return 'secondary'
                    except:
                        pass
                
                # Fallback: assume primary
                self.storage_map_cache[schema_name] = 'primary'
                return 'primary'
                
        except Exception as e:
            # Fallback to primary on any error
            self.storage_map_cache[schema_name] = 'primary'
            return 'primary'
    
    def get_schemas_by_location(self):
        """Get schemas grouped by their storage location."""
        self.update_storage_map()
        
        locations = {
            'primary': [],
            'secondary': [],
            'secondary_only': [],
            'unknown': []
        }
        
        for schema, location in self.storage_map_cache.items():
            if location in locations:
                locations[location].append(schema)
            else:
                locations['unknown'].append(schema)
        
        return locations
    
    def get_all_databases_enhanced(self, cursor):
        """Enhanced version of get_all_databases that includes storage location info."""
        # Get the original database list
        databases = get_all_databases(cursor)
        
        # Add storage location information
        enhanced_databases = []
        for db_name in databases:
            location = self.get_schema_location(db_name)
            creation_time = extract_timestamp_from_name(db_name)
            
            enhanced_databases.append({
                'name': db_name,
                'location': location,
                'creation_time': creation_time.isoformat() if creation_time else None,
                'age_days': (datetime.now() - creation_time).days if creation_time else None
            })
        
        return enhanced_databases
    
    def check_schema_accessibility(self, schema_name):
        """Check if a schema is accessible regardless of its storage location."""
        try:
            connection = create_connection()
            cursor = connection.cursor()
            
            # Check if database exists
            cursor.execute("SHOW DATABASES LIKE %s", (schema_name,))
            result = cursor.fetchone()
            
            if result:
                # Try to access a table in the database
                cursor.execute(f"USE `{schema_name}`")
                cursor.execute("SHOW TABLES LIMIT 1")
                tables = cursor.fetchall()
                
                cursor.close()
                connection.close()
                
                return {
                    'accessible': True,
                    'location': self.get_schema_location(schema_name),
                    'table_count': len(tables)
                }
            else:
                cursor.close()
                connection.close()
                return {
                    'accessible': False,
                    'location': self.get_schema_location(schema_name),
                    'error': 'Database not found'
                }
        
        except Exception as e:
            return {
                'accessible': False,
                'location': self.get_schema_location(schema_name),
                'error': str(e)
            }

# Global instance for the webapp to use
dual_storage = DualStorageManager()

def get_all_databases_with_storage_info(cursor):
    """Wrapper function for backward compatibility with enhanced info."""
    return dual_storage.get_all_databases_enhanced(cursor)

def get_storage_statistics():
    """Get comprehensive storage statistics for both locations."""
    schema_sizes = get_schema_sizes()  # From original db_operations
    storage_info = dual_storage.get_storage_info()
    schemas_by_location = dual_storage.get_schemas_by_location()
    
    # Calculate storage usage by location
    location_stats = {
        'primary': {'count': 0, 'total_size': 0},
        'secondary': {'count': 0, 'total_size': 0},
        'secondary_only': {'count': 0, 'total_size': 0}
    }
    
    # Handle case where schema_sizes might be a list or dict
    if isinstance(schema_sizes, list):
        # Convert list to dict format - handle different possible structures
        schema_sizes_dict = {}
        for item in schema_sizes:
            if isinstance(item, dict):
                # Handle dict format like {'schema_name': 'name', 'size_bytes': 123}
                if 'schema_name' in item and 'size_bytes' in item:
                    schema_sizes_dict[item['schema_name']] = item['size_bytes']
                elif 'name' in item and 'size' in item:
                    schema_sizes_dict[item['name']] = item['size']
            else:
                # Handle other formats
                print(f"Unexpected schema_sizes item format: {type(item)}, {item}")
    else:
        schema_sizes_dict = schema_sizes
    
    for schema_name, size in schema_sizes_dict.items():
        location = dual_storage.get_schema_location(schema_name)
        if location in location_stats:
            location_stats[location]['count'] += 1
            location_stats[location]['total_size'] += size
    
    return {
        'storage_info': storage_info,
        'location_stats': location_stats,
        'schemas_by_location': schemas_by_location
    }

def test_dual_storage_access():
    """Test function to verify dual storage access is working."""
    print("🧪 Testing Dual Storage Access")
    print("=" * 50)
    
    try:
        # Test database connection
        connection = create_connection()
        cursor = connection.cursor()
        
        # Get enhanced database list
        enhanced_databases = get_all_databases_with_storage_info(cursor)
        
        print(f"✅ Found {len(enhanced_databases)} databases")
        
        # Test storage statistics
        stats = get_storage_statistics()
        
        print(f"📊 Storage Statistics:")
        for location, stat in stats['location_stats'].items():
            if stat['count'] > 0:
                size_str = format_size(stat['total_size'])
                print(f"   {location}: {stat['count']} schemas, {size_str}")
        
        # Test a few schema access
        test_schemas = enhanced_databases[:5]  # Test first 5 schemas
        
        print(f"\n🔍 Testing schema accessibility:")
        for db_info in test_schemas:
            schema_name = db_info['name']
            access_info = dual_storage.check_schema_accessibility(schema_name)
            
            status = "✅" if access_info['accessible'] else "❌"
            location = access_info['location']
            print(f"   {status} {schema_name} ({location})")
        
        cursor.close()
        connection.close()
        
        print("\n✅ Dual storage access test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Dual storage access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test the dual storage system."""
    return test_dual_storage_access()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)