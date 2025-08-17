#!/usr/bin/env python3
"""
Generate a comprehensive summary of the current storage allocation status.
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from db_operations import get_schema_storage_type, extract_timestamp_from_schema_name
from datetime import datetime, timedelta
import mysql.connector

def generate_storage_summary():
    """Generate a comprehensive storage summary."""
    print("🎯 STORAGE ALLOCATION STATUS SUMMARY")
    print("=" * 60)
    
    # Get all schemas
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
    except Exception as e:
        print(f"Error getting schemas: {e}")
        return
    
    # Categorize schemas
    cutoff_date = datetime.now() - timedelta(days=60)
    
    recent_primary = 0      # Recent schemas on primary (CORRECT)
    recent_sda1 = 0         # Recent schemas on SDA1 (INCORRECT)
    old_primary = 0         # Old schemas on primary (NEEDS MIGRATION)
    old_sda1 = 0           # Old schemas on SDA1 (CORRECT)
    no_timestamp = 0        # Schemas without timestamp
    not_found = 0          # Schemas not found in filesystem
    
    for schema in databases:
        timestamp = extract_timestamp_from_schema_name(schema)
        storage_info = get_schema_storage_type(schema)
        
        if timestamp:
            age_days = (datetime.now() - timestamp).days
            is_old = age_days >= 60
            
            if storage_info['type'] == 'primary':
                if is_old:
                    old_primary += 1
                else:
                    recent_primary += 1
            elif storage_info['type'] == 'archive_with_direct_access':
                if is_old:
                    old_sda1 += 1
                else:
                    recent_sda1 += 1
            elif storage_info['type'] == 'not_found':
                not_found += 1
        else:
            no_timestamp += 1
    
    total_schemas = len(databases)
    total_old = old_primary + old_sda1
    total_recent = recent_primary + recent_sda1
    
    print(f"📊 OVERVIEW:")
    print(f"   Total schemas: {total_schemas}")
    print(f"   Recent (< 60 days): {total_recent}")
    print(f"   Old (≥ 60 days): {total_old}")
    print(f"   No timestamp: {no_timestamp}")
    print(f"   Not found: {not_found}")
    
    print(f"\n🎯 ALLOCATION ACCURACY:")
    print(f"   ✅ Recent schemas on PRIMARY: {recent_primary} (CORRECT)")
    print(f"   ❌ Recent schemas on SDA1: {recent_sda1} (INCORRECT)")
    print(f"   ❌ Old schemas on PRIMARY: {old_primary} (NEEDS MIGRATION)")
    print(f"   ✅ Old schemas on SDA1: {old_sda1} (CORRECT)")
    
    # Calculate percentages
    if total_recent > 0:
        recent_accuracy = (recent_primary / total_recent) * 100
    else:
        recent_accuracy = 100
        
    if total_old > 0:
        old_accuracy = (old_sda1 / total_old) * 100
    else:
        old_accuracy = 100
    
    overall_accuracy = ((recent_primary + old_sda1) / (total_recent + total_old)) * 100 if (total_recent + total_old) > 0 else 100
    
    print(f"\n📈 ACCURACY METRICS:")
    print(f"   Recent schema accuracy: {recent_accuracy:.1f}%")
    print(f"   Old schema accuracy: {old_accuracy:.1f}%")
    print(f"   Overall accuracy: {overall_accuracy:.1f}%")
    
    print(f"\n💾 STORAGE DISTRIBUTION:")
    primary_total = recent_primary + old_primary
    sda1_total = recent_sda1 + old_sda1
    print(f"   📁 Primary storage: {primary_total} schemas ({primary_total/total_schemas*100:.1f}%)")
    print(f"   📦 SDA1 storage: {sda1_total} schemas ({sda1_total/total_schemas*100:.1f}%)")
    
    if old_primary > 0:
        print(f"\n⚠️  ACTION NEEDED:")
        print(f"   {old_primary} old schemas need migration to SDA1")
        print(f"   Run: python3 migrate_remaining_old_schemas.py")
    else:
        print(f"\n🎉 PERFECT ALLOCATION!")
        print(f"   All schemas are correctly allocated based on age")
    
    print(f"\n🔗 SYMLINK SUMMARY:")
    print(f"   {old_sda1} schemas use direct SDA1 access via symlinks")
    print(f"   This provides instant access to archived data!")

if __name__ == "__main__":
    generate_storage_summary()