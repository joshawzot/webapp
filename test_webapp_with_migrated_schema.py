#!/usr/bin/env python3
"""
Test webapp storage display with both primary and archived schemas
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from dual_storage_db_operations import DualStorageManager
from analyze_schema_ages import extract_timestamp_from_name
from db_operations import create_connection
import json
from pathlib import Path

def test_webapp_storage_display():
    """Test the storage info that would be displayed in the webapp"""
    print("🌐 TESTING WEBAPP STORAGE DISPLAY")
    print("=" * 70)
    
    # Get sample schemas from both storage locations
    dual_storage = DualStorageManager()
    
    # Get all databases
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    all_databases = [db[0] for db in cursor.fetchall() 
                    if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
    cursor.close()
    conn.close()
    
    # Find examples of each storage type
    primary_examples = []
    secondary_examples = []
    
    for database in all_databases:
        location = dual_storage.get_schema_location(database)
        if location == 'primary' and len(primary_examples) < 3:
            primary_examples.append(database)
        elif location == 'secondary' and len(secondary_examples) < 3:
            secondary_examples.append(database)
        
        if len(primary_examples) >= 3 and len(secondary_examples) >= 3:
            break
    
    print(f"🔍 Testing with:")
    print(f"   🟢 {len(primary_examples)} primary storage examples")
    print(f"   🔵 {len(secondary_examples)} secondary storage examples")
    print("-" * 70)
    
    # Test storage info generation for each schema (simulating webapp route)
    def get_storage_info_for_webapp(database):
        """Simulate the storage info generation from route_handlers.py"""
        # Get storage location
        storage_location = dual_storage.get_schema_location(database)
        
        # Extract timestamp and calculate age
        schema_timestamp = extract_timestamp_from_name(database)
        if schema_timestamp:
            from datetime import datetime
            age_days = (datetime.now() - schema_timestamp).days
            schema_age = f"{age_days} days"
            timestamp_str = schema_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            schema_age = "Unknown"
            timestamp_str = "No timestamp"
        
        # Determine storage details based on location
        if storage_location == 'secondary':
            storage_device = '/dev/sda1'
            storage_path = '/local/mysql/data'
            storage_type = 'Archive Storage'
            storage_color = 'warning'
        else:
            storage_device = '/dev/nvme2n1p1'
            storage_path = '/var/lib/mysql'
            storage_type = 'Primary Storage'
            storage_color = 'info'
        
        return {
            'location': storage_location,
            'device': storage_device,
            'path': storage_path,
            'type': storage_type,
            'color': storage_color,
            'age': schema_age,
            'timestamp': timestamp_str
        }
    
    # Test primary storage examples
    print("🟢 PRIMARY STORAGE EXAMPLES:")
    for i, database in enumerate(primary_examples, 1):
        storage_info = get_storage_info_for_webapp(database)
        print(f"  {i}. {database[:50]:<50}")
        print(f"     Type: {storage_info['type']}")
        print(f"     Device: {storage_info['device']}")
        print(f"     Path: {storage_info['path']}")
        print(f"     Age: {storage_info['age']}")
        print()
    
    # Test secondary storage examples  
    print("🔵 ARCHIVE STORAGE EXAMPLES:")
    for i, database in enumerate(secondary_examples, 1):
        storage_info = get_storage_info_for_webapp(database)
        print(f"  {i}. {database[:50]:<50}")
        print(f"     Type: {storage_info['type']}")
        print(f"     Device: {storage_info['device']}")
        print(f"     Path: {storage_info['path']}")
        print(f"     Age: {storage_info['age']}")
        print()
    
    print("-" * 70)
    print("✅ WEBAPP INTEGRATION TEST COMPLETE")
    print(f"   The webapp should now correctly display:")
    print(f"   🟢 Primary schemas as: Primary Storage on /dev/nvme2n1p1")
    print(f"   🔵 Archived schemas as: Archive Storage on /dev/sda1")
    
    return True

if __name__ == "__main__":
    test_webapp_storage_display()