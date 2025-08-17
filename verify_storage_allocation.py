#!/usr/bin/env python3
"""
Verify that storage allocation is working correctly based on schema age.
Schemas older than 60 days should be on SDA1, newer ones on primary storage.
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import get_schema_storage_type, extract_timestamp_from_schema_name
from datetime import datetime, timedelta
import mysql.connector

def get_all_schemas():
    """Get all schemas from MySQL."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        cursor = connection.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall() 
                    if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        cursor.close()
        connection.close()
        return databases
    except Exception as e:
        print(f"Error getting schemas: {e}")
        return []

def analyze_storage_allocation():
    """Analyze storage allocation for all schemas."""
    print("🔍 ANALYZING STORAGE ALLOCATION")
    print("=" * 60)
    
    schemas = get_all_schemas()
    print(f"📋 Found {len(schemas)} schemas total")
    
    # Categorize schemas
    recent_schemas = []      # Should be on primary
    old_schemas = []         # Should be on SDA1 (symlinked)
    no_timestamp = []        # No timestamp in name
    
    cutoff_date = datetime.now() - timedelta(days=60)
    print(f"📅 Cutoff date (60 days ago): {cutoff_date.strftime('%Y-%m-%d')}")
    print()
    
    # Analyze each schema
    for schema in schemas[:20]:  # Test first 20 schemas
        timestamp = extract_timestamp_from_schema_name(schema)
        storage_info = get_schema_storage_type(schema)
        
        if timestamp:
            age_days = (datetime.now() - timestamp).days
            if timestamp < cutoff_date:
                # Should be on SDA1
                old_schemas.append({
                    'name': schema,
                    'age_days': age_days,
                    'timestamp': timestamp,
                    'storage_type': storage_info['type'],
                    'display': storage_info['display'],
                    'is_symlink': storage_info['is_symlink']
                })
            else:
                # Should be on primary
                recent_schemas.append({
                    'name': schema,
                    'age_days': age_days,
                    'timestamp': timestamp,
                    'storage_type': storage_info['type'],
                    'display': storage_info['display'],
                    'is_symlink': storage_info['is_symlink']
                })
        else:
            no_timestamp.append({
                'name': schema,
                'storage_type': storage_info['type'],
                'display': storage_info['display'],
                'is_symlink': storage_info['is_symlink']
            })
    
    # Report results
    print("📊 STORAGE ALLOCATION RESULTS:")
    print("=" * 60)
    
    print(f"\n🟢 RECENT SCHEMAS (< 60 days old) - Should be on PRIMARY:")
    print(f"   Count: {len(recent_schemas)}")
    for schema in recent_schemas[:5]:  # Show first 5
        status = "✅" if schema['storage_type'] == 'primary' else "❌"
        print(f"   {status} {schema['name'][:50]}...")
        print(f"      Age: {schema['age_days']} days | Storage: {schema['display']}")
    
    print(f"\n🔵 OLD SCHEMAS (≥ 60 days old) - Should be on SDA1:")
    print(f"   Count: {len(old_schemas)}")
    for schema in old_schemas[:5]:  # Show first 5
        status = "✅" if schema['storage_type'] == 'archive_with_direct_access' else "❌"
        print(f"   {status} {schema['name'][:50]}...")
        print(f"      Age: {schema['age_days']} days | Storage: {schema['display']}")
    
    print(f"\n⚪ NO TIMESTAMP SCHEMAS:")
    print(f"   Count: {len(no_timestamp)}")
    for schema in no_timestamp[:3]:  # Show first 3
        print(f"   • {schema['name'][:50]}...")
        print(f"      Storage: {schema['display']}")
    
    # Summary
    recent_on_primary = sum(1 for s in recent_schemas if s['storage_type'] == 'primary')
    old_on_sda1 = sum(1 for s in old_schemas if s['storage_type'] == 'archive_with_direct_access')
    
    print(f"\n📈 SUMMARY:")
    print(f"   🟢 Recent schemas on primary: {recent_on_primary}/{len(recent_schemas)}")
    print(f"   🔵 Old schemas on SDA1: {old_on_sda1}/{len(old_schemas)}")
    
    if recent_on_primary == len(recent_schemas) and old_on_sda1 == len(old_schemas):
        print(f"\n🎉 PERFECT! All schemas are correctly allocated!")
    else:
        print(f"\n⚠️  Some schemas need attention:")
        if recent_on_primary < len(recent_schemas):
            print(f"   - {len(recent_schemas) - recent_on_primary} recent schemas not on primary")
        if old_on_sda1 < len(old_schemas):
            print(f"   - {len(old_schemas) - old_on_sda1} old schemas not properly symlinked to SDA1")

if __name__ == "__main__":
    analyze_storage_allocation()