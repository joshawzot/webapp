#!/usr/bin/env python3
"""
Test webapp storage display functionality
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from dual_storage_db_operations import dual_storage
from analyze_schema_ages import extract_timestamp_from_name
from db_operations import create_connection

def test_storage_display():
    """Test the storage display functionality"""
    print("🧪 TESTING WEBAPP STORAGE DISPLAY")
    print("=" * 50)
    
    try:
        # Get a few example databases
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall() 
                    if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        cursor.close()
        conn.close()
        
        # Test with first few databases
        test_databases = databases[:3]
        
        print(f"📊 Testing storage info for {len(test_databases)} databases:")
        print("-" * 80)
        
        for database in test_databases:
            storage_location = dual_storage.get_schema_location(database)
            schema_timestamp = extract_timestamp_from_name(database)
            
            # Determine storage device and path
            if storage_location == 'secondary' or 'migrated' in storage_location:
                storage_device = "/dev/sda1 (Archive Drive)"
                storage_path = "/local/mysql/data"
                storage_type = "Archive Storage"
                storage_color = "info"
            else:
                storage_device = "/dev/nvme2n1p1 (Primary Drive)"
                storage_path = "/var/lib/mysql"
                storage_type = "Primary Storage"
                storage_color = "success"
            
            # Calculate age if timestamp exists
            schema_age = None
            if schema_timestamp:
                from datetime import datetime
                age_days = (datetime.now() - schema_timestamp).days
                schema_age = f"{age_days} days old"
            
            print(f"🗄️  Database: {database}")
            print(f"   📁 Storage: {storage_type}")
            print(f"   💾 Device: {storage_device}")
            print(f"   📂 Path: {storage_path}")
            if schema_age:
                print(f"   📅 Age: {schema_age}")
            if schema_timestamp:
                print(f"   🕐 Created: {schema_timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"   🎨 Color: {storage_color}")
            print("-" * 80)
        
        print("✅ Storage display test completed successfully!")
        print("\n🌐 Now you can test in the webapp:")
        print("1. Go to your webapp homepage")
        print("2. Select any database")
        print("3. Check the storage information display below the schema name")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_storage_display()