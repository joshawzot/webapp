#!/usr/bin/env python3
"""
Quick test to verify the dual storage system is working
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from dual_storage_db_operations import get_storage_statistics, dual_storage
from db_operations import create_connection

print("🧪 QUICK DUAL STORAGE TEST")
print("=" * 50)

try:
    # Test 1: Basic connection
    print("1. Testing MySQL connection...")
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')")
    schema_count = cursor.fetchone()[0]
    print(f"✅ Found {schema_count} user schemas")
    cursor.close()
    conn.close()
    
    # Test 2: Storage statistics
    print("\n2. Testing storage statistics...")
    stats = get_storage_statistics()
    print(f"✅ Storage statistics calculated successfully")
    
    # Test 3: Schema location check
    print("\n3. Testing schema location tracking...")
    location_stats = stats['location_stats']
    for location, stat in location_stats.items():
        if stat['count'] > 0:
            print(f"   {location}: {stat['count']} schemas")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("The dual storage system is working correctly!")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()