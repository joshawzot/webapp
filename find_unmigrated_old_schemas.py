#!/usr/bin/env python3
"""
Find old schemas (>60 days) that are still on primary storage 
and should be migrated to SDA1 via symlinks.
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import get_schema_storage_type, extract_timestamp_from_schema_name
from datetime import datetime, timedelta
import mysql.connector
import subprocess

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

def find_unmigrated_old_schemas():
    """Find old schemas that should be migrated to SDA1."""
    print("🔍 FINDING UNMIGRATED OLD SCHEMAS")
    print("=" * 50)
    
    schemas = get_all_schemas()
    cutoff_date = datetime.now() - timedelta(days=60)
    
    unmigrated_old_schemas = []
    migrated_old_schemas = []
    recent_schemas = []
    
    print(f"📅 Cutoff date (60 days ago): {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"📋 Checking {len(schemas)} total schemas...")
    
    for schema in schemas:
        timestamp = extract_timestamp_from_schema_name(schema)
        if timestamp:
            age_days = (datetime.now() - timestamp).days
            storage_info = get_schema_storage_type(schema)
            
            if age_days >= 60:
                # This is an old schema
                if storage_info['type'] == 'primary':
                    # Old schema still on primary - needs migration
                    unmigrated_old_schemas.append({
                        'name': schema,
                        'age_days': age_days,
                        'timestamp': timestamp
                    })
                elif storage_info['type'] == 'archive_with_direct_access':
                    # Old schema properly migrated
                    migrated_old_schemas.append({
                        'name': schema,
                        'age_days': age_days,
                        'timestamp': timestamp
                    })
            else:
                # Recent schema
                if storage_info['type'] == 'primary':
                    recent_schemas.append(schema)
    
    # Report results
    print(f"\n📊 RESULTS:")
    print(f"🟢 Recent schemas (< 60 days) on primary: {len(recent_schemas)}")
    print(f"🔵 Old schemas (≥ 60 days) properly migrated to SDA1: {len(migrated_old_schemas)}")
    print(f"❌ Old schemas (≥ 60 days) still on primary (NEED MIGRATION): {len(unmigrated_old_schemas)}")
    
    if unmigrated_old_schemas:
        print(f"\n🚨 SCHEMAS THAT NEED MIGRATION TO SDA1:")
        print("=" * 50)
        for i, schema in enumerate(unmigrated_old_schemas[:10], 1):  # Show first 10
            print(f"{i:2d}. {schema['name']}")
            print(f"    Age: {schema['age_days']} days ({schema['timestamp'].strftime('%Y-%m-%d')})")
        
        if len(unmigrated_old_schemas) > 10:
            print(f"    ... and {len(unmigrated_old_schemas) - 10} more")
        
        # Estimate space that could be freed
        print(f"\n💾 POTENTIAL SPACE SAVINGS:")
        print(f"   {len(unmigrated_old_schemas)} schemas could be moved to SDA1")
        print(f"   This would free up primary storage space")
        
        return unmigrated_old_schemas
    else:
        print(f"\n🎉 ALL OLD SCHEMAS ARE PROPERLY MIGRATED!")
        return []

if __name__ == "__main__":
    unmigrated = find_unmigrated_old_schemas()
    
    if unmigrated:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Run a migration script to move these {len(unmigrated)} old schemas")
        print(f"   from primary storage to SDA1 via symlinks.")