#!/usr/bin/env python3
"""
Get Schema Creation Dates
Quick utility to list schemas with their creation dates for verification.
"""

import sys
sys.path.append('/home/admin2/webapp_2')
from analyze_schema_ages import extract_timestamp_from_name, get_schema_sizes, format_size
from datetime import datetime, timedelta

def main():
    """List all schemas with their creation dates."""
    print("Schema Creation Dates Report")
    print("=" * 80)
    
    schema_sizes = get_schema_sizes()
    cutoff_date = datetime.now() - timedelta(days=60)
    
    # Categorize schemas
    old_schemas = []
    new_schemas = []
    no_timestamp = []
    
    for schema_name, size in schema_sizes.items():
        creation_time = extract_timestamp_from_name(schema_name)
        
        if creation_time is None:
            no_timestamp.append((schema_name, size))
        elif creation_time < cutoff_date:
            old_schemas.append((schema_name, size, creation_time))
        else:
            new_schemas.append((schema_name, size, creation_time))
    
    # Sort by creation time
    old_schemas.sort(key=lambda x: x[2])
    new_schemas.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n📊 SUMMARY:")
    print(f"Old schemas (>60 days): {len(old_schemas)}")
    print(f"New schemas (≤60 days): {len(new_schemas)}")
    print(f"No timestamp: {len(no_timestamp)}")
    
    print(f"\n🔴 OLD SCHEMAS (first 20):")
    for i, (name, size, creation_time) in enumerate(old_schemas[:20]):
        age_days = (datetime.now() - creation_time).days
        print(f"{i+1:3d}. {name:<60} {format_size(size):<12} {creation_time.strftime('%Y-%m-%d %H:%M')} ({age_days} days)")
    
    if len(old_schemas) > 20:
        print(f"... and {len(old_schemas) - 20} more old schemas")
    
    print(f"\n🟢 NEW SCHEMAS (first 10):")
    for i, (name, size, creation_time) in enumerate(new_schemas[:10]):
        age_days = (datetime.now() - creation_time).days
        print(f"{i+1:3d}. {name:<60} {format_size(size):<12} {creation_time.strftime('%Y-%m-%d %H:%M')} ({age_days} days)")

if __name__ == "__main__":
    main()