#!/usr/bin/env python3
"""
Schema Age Analysis Script
Analyzes MySQL schemas to identify those created more than 60 days ago based on timestamp in schema names.
"""

import mysql.connector
from datetime import datetime, timedelta
import re
import sys
import os

# Import database configuration from the main application
sys.path.append('/home/admin2/webapp_2')
from db_operations import DB_CONFIG, create_connection

def extract_timestamp_from_name(db_name):
    """Extract timestamp from database name if it exists."""
    # Look for patterns like _YYYYMMDDHHMMSS
    match = re.search(r'_(\d{14})(?:$|_)', db_name)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        except ValueError:
            pass
    
    # Look for patterns like _YYYYMMDD
    match = re.search(r'_(\d{8})(?:$|_)', db_name)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d')
        except ValueError:
            pass
    
    return None

def get_schema_sizes():
    """Get all schemas/databases with their sizes."""
    connection = create_connection()
    cursor = connection.cursor()
    
    try:
        # Define restricted databases to exclude
        restricted_dbs = ['performance_schema', 'mysql', 'information_schema', 'sys']
        
        # Query to get individual database sizes
        query = """
        SELECT 
            table_schema as 'schema_name',
            SUM(data_length + index_length) as 'total_size'
        FROM information_schema.TABLES
        WHERE table_schema NOT IN ({})
        GROUP BY table_schema
        ORDER BY total_size DESC
        """.format(','.join(['%s'] * len(restricted_dbs)))
        
        cursor.execute(query, restricted_dbs)
        results = cursor.fetchall()
        
        # Format the results
        schema_sizes = {}
        for schema_name, total_size in results:
            schema_sizes[schema_name] = total_size if total_size else 0
        
        return schema_sizes
        
    except mysql.connector.Error as err:
        print(f"Error calculating schema sizes: {err}")
        return {}
    finally:
        cursor.close()
        connection.close()

def format_size(size_bytes):
    """Convert size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def analyze_schemas():
    """Analyze all schemas and categorize them by age."""
    print("Analyzing MySQL schemas for age-based migration...")
    print("=" * 80)
    
    # Get current date for comparison
    current_date = datetime.now()
    cutoff_date = current_date - timedelta(days=60)
    
    print(f"Current date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cutoff date (60 days ago): {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Get schema sizes
    schema_sizes = get_schema_sizes()
    
    # Categorize schemas
    old_schemas = []  # > 60 days old
    new_schemas = []  # <= 60 days old
    no_timestamp_schemas = []  # No timestamp in name
    
    total_old_size = 0
    total_new_size = 0
    total_no_timestamp_size = 0
    
    for schema_name, size in schema_sizes.items():
        creation_time = extract_timestamp_from_name(schema_name)
        
        if creation_time is None:
            no_timestamp_schemas.append({
                'name': schema_name,
                'size': size,
                'formatted_size': format_size(size)
            })
            total_no_timestamp_size += size
        elif creation_time < cutoff_date:
            old_schemas.append({
                'name': schema_name,
                'creation_time': creation_time,
                'size': size,
                'formatted_size': format_size(size),
                'age_days': (current_date - creation_time).days
            })
            total_old_size += size
        else:
            new_schemas.append({
                'name': schema_name,
                'creation_time': creation_time,
                'size': size,
                'formatted_size': format_size(size),
                'age_days': (current_date - creation_time).days
            })
            total_new_size += size
    
    # Sort by age for better readability
    old_schemas.sort(key=lambda x: x['creation_time'])
    new_schemas.sort(key=lambda x: x['creation_time'], reverse=True)
    
    # Print results
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Total schemas analyzed: {len(schema_sizes)}")
    print(f"Schemas to migrate to /dev/sda1 (>60 days): {len(old_schemas)}")
    print(f"Schemas to keep on /dev/nvme0n1p3 (≤60 days): {len(new_schemas)}")
    print(f"Schemas without timestamp (skipped): {len(no_timestamp_schemas)}")
    print("-" * 80)
    
    print(f"\n🔴 SCHEMAS TO MIGRATE TO /dev/sda1 ({len(old_schemas)} schemas, {format_size(total_old_size)}):")
    if old_schemas:
        print(f"{'Schema Name':<60} {'Age (days)':<12} {'Size':<15} {'Created':<20}")
        print("-" * 107)
        for schema in old_schemas:
            print(f"{schema['name']:<60} {schema['age_days']:<12} {schema['formatted_size']:<15} {schema['creation_time'].strftime('%Y-%m-%d %H:%M'):<20}")
    else:
        print("No old schemas found!")
    
    print(f"\n🟢 SCHEMAS TO KEEP ON /dev/nvme0n1p3 ({len(new_schemas)} schemas, {format_size(total_new_size)}):")
    if new_schemas:
        print(f"{'Schema Name':<60} {'Age (days)':<12} {'Size':<15} {'Created':<20}")
        print("-" * 107)
        for schema in new_schemas[:10]:  # Show only first 10 for brevity
            print(f"{schema['name']:<60} {schema['age_days']:<12} {schema['formatted_size']:<15} {schema['creation_time'].strftime('%Y-%m-%d %H:%M'):<20}")
        if len(new_schemas) > 10:
            print(f"... and {len(new_schemas) - 10} more recent schemas")
    else:
        print("No recent schemas found!")
    
    print(f"\n⚪ SCHEMAS WITHOUT TIMESTAMP (SKIPPED) ({len(no_timestamp_schemas)} schemas, {format_size(total_no_timestamp_size)}):")
    if no_timestamp_schemas:
        print(f"{'Schema Name':<60} {'Size':<15}")
        print("-" * 75)
        for schema in no_timestamp_schemas:
            print(f"{schema['name']:<60} {schema['formatted_size']:<15}")
    
    print(f"\n💾 STORAGE IMPACT:")
    print(f"Data to move to /dev/sda1: {format_size(total_old_size)}")
    print(f"Data to keep on /dev/nvme0n1p3: {format_size(total_new_size)}")
    print(f"Total data size: {format_size(total_old_size + total_new_size + total_no_timestamp_size)}")
    
    # Return the categorized data for use by other scripts
    return {
        'old_schemas': old_schemas,
        'new_schemas': new_schemas,
        'no_timestamp_schemas': no_timestamp_schemas,
        'cutoff_date': cutoff_date,
        'total_old_size': total_old_size,
        'total_new_size': total_new_size,
        'total_no_timestamp_size': total_no_timestamp_size
    }

def main():
    """Main function to run the analysis."""
    try:
        analysis_results = analyze_schemas()
        
        # Save results for use by migration script
        output_file = 'schema_analysis_results.txt'
        with open(output_file, 'w') as f:
            f.write("Schema Analysis Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cutoff Date: {analysis_results['cutoff_date'].strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Schemas to migrate to /dev/sda1 (>60 days):\n")
            for schema in analysis_results['old_schemas']:
                f.write(f"  {schema['name']} - {schema['formatted_size']} - {schema['age_days']} days old\n")
            
            f.write(f"\nTotal schemas to migrate: {len(analysis_results['old_schemas'])}\n")
            f.write(f"Total size to migrate: {format_size(analysis_results['total_old_size'])}\n")
        
        print(f"\n📄 Analysis results saved to: {output_file}")
        
        # Ask user if they want to proceed with migration setup
        if analysis_results['old_schemas']:
            print(f"\n🚀 Ready to proceed with migration setup?")
            print(f"This will affect {len(analysis_results['old_schemas'])} schemas totaling {format_size(analysis_results['total_old_size'])}")
        else:
            print(f"\n✅ No schemas older than 60 days found. No migration needed at this time.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())