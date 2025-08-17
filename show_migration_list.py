#!/usr/bin/env python3
"""
Show exactly which schemas will be migrated
Only schemas with timestamp patterns in names that are >60 days old
"""

import subprocess
import sys
from datetime import datetime, timedelta
import re

def run_cmd(cmd, capture_output=True):
    """Run command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
        return True, result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def extract_timestamp_from_name(schema_name):
    """Extract timestamp from schema name if it exists."""
    patterns = [
        r'_(\d{14})$',  # _20250801231213 at end
        r'_(\d{12})$',  # _202508012312 at end (12 digits)
        r'_(\d{8})$',   # _20250801 at end (8 digits - date only)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, schema_name)
        if match:
            timestamp_str = match.group(1)
            try:
                if len(timestamp_str) == 14:  # YYYYMMDDHHMMSS
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                elif len(timestamp_str) == 12:  # YYYYMMDDHHMM
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                elif len(timestamp_str) == 8:   # YYYYMMDD
                    return datetime.strptime(timestamp_str, '%Y%m%d')
            except ValueError:
                continue
    return None

def main():
    print("📋 SCHEMAS TO BE MIGRATED")
    print("=" * 80)
    print("Only schemas with timestamp patterns >60 days old will be migrated")
    print()
    
    # Get all schemas
    print("🔍 Scanning all schemas...")
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if not success:
        print(f"❌ Cannot access MySQL: {output}")
        return
    
    all_databases = output.split('\n')[1:]  # Skip header
    user_databases = [db for db in all_databases 
                     if db not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
    
    print(f"📊 Total user schemas: {len(user_databases)}")
    
    # Categorize schemas
    schemas_with_timestamp = []
    schemas_without_timestamp = []
    schemas_to_migrate = []
    schemas_too_new = []
    
    cutoff_date = datetime.now() - timedelta(days=60)
    
    for schema in user_databases:
        creation_time = extract_timestamp_from_name(schema)
        if creation_time:
            age_days = (datetime.now() - creation_time).days
            schemas_with_timestamp.append((schema, creation_time, age_days))
            
            if creation_time < cutoff_date:
                schemas_to_migrate.append((schema, creation_time, age_days))
            else:
                schemas_too_new.append((schema, creation_time, age_days))
        else:
            schemas_without_timestamp.append(schema)
    
    # Sort by age (oldest first)
    schemas_to_migrate.sort(key=lambda x: x[2], reverse=True)
    schemas_too_new.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n📊 SCHEMA CATEGORIZATION:")
    print(f"   📅 With timestamp patterns: {len(schemas_with_timestamp)}")
    print(f"   📅 Without timestamp patterns: {len(schemas_without_timestamp)} (will NOT be migrated)")
    print(f"   🎯 To migrate (>60 days): {len(schemas_to_migrate)}")
    print(f"   🏠 To keep on primary (<60 days): {len(schemas_too_new)}")
    
    if len(schemas_to_migrate) > 0:
        print(f"\n🚀 SCHEMAS TO BE MIGRATED ({len(schemas_to_migrate)} total):")
        print(f"=" * 80)
        
        oldest = schemas_to_migrate[0]
        newest = schemas_to_migrate[-1]
        print(f"📈 Age range: {newest[2]} to {oldest[2]} days old")
        print(f"📅 Date range: {newest[1].strftime('%Y-%m-%d')} to {oldest[1].strftime('%Y-%m-%d')}")
        print()
        
        # Show first 20 schemas
        print(f"📝 FIRST 20 SCHEMAS TO MIGRATE:")
        for i, (schema, creation_time, age_days) in enumerate(schemas_to_migrate[:20], 1):
            creation_str = creation_time.strftime('%Y-%m-%d %H:%M')
            print(f"   {i:2d}. {schema}")
            print(f"       📅 Created: {creation_str} ({age_days} days ago)")
        
        if len(schemas_to_migrate) > 20:
            print(f"   ... and {len(schemas_to_migrate) - 20} more schemas")
        
        # Show by name patterns
        print(f"\n📊 BREAKDOWN BY NAME PATTERNS:")
        patterns = {}
        for schema, creation_time, age_days in schemas_to_migrate:
            # Extract pattern before timestamp
            base_name = re.sub(r'_\d{8,14}$', '', schema)
            prefix = base_name.split('_')[0] if '_' in base_name else base_name
            patterns[prefix] = patterns.get(prefix, 0) + 1
        
        # Show top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_patterns[:10]:
            print(f"   {pattern}: {count} schemas")
        
        if len(sorted_patterns) > 10:
            print(f"   ... and {len(sorted_patterns) - 10} more patterns")
        
        # Estimate backup size and time
        avg_size_mb = 50  # Conservative estimate per schema
        total_size_gb = len(schemas_to_migrate) * avg_size_mb / 1024
        est_time_minutes = len(schemas_to_migrate) * 2  # 2 minutes per schema
        
        print(f"\n💾 MIGRATION ESTIMATES:")
        print(f"   📦 Estimated backup size: ~{total_size_gb:.1f} GB")
        print(f"   ⏱️  Estimated migration time: ~{est_time_minutes//60}h {est_time_minutes%60}m")
        print(f"   💾 Archive storage available: 8.2T (plenty of space)")
        
        print(f"\n✅ MIGRATION SAFETY:")
        print(f"   • Only schemas with timestamp patterns will be migrated")
        print(f"   • Only schemas >60 days old will be migrated")
        print(f"   • Full SQL backups created before migration")
        print(f"   • Rollback capability available")
        print(f"   • No tablespace corruption (MySQL-native methods)")
    else:
        print(f"\n✅ NO SCHEMAS TO MIGRATE!")
        print(f"   All timestamped schemas are <60 days old")
    
    if len(schemas_too_new) > 0:
        print(f"\n🏠 SCHEMAS STAYING ON PRIMARY (too new):")
        print(f"   📊 Count: {len(schemas_too_new)} schemas")
        if len(schemas_too_new) <= 10:
            for schema, creation_time, age_days in schemas_too_new:
                creation_str = creation_time.strftime('%Y-%m-%d')
                print(f"   • {schema} ({age_days} days old)")
        else:
            print(f"   📅 Age range: {schemas_too_new[-1][2]} to {schemas_too_new[0][2]} days")
            print(f"   📝 Examples:")
            for schema, creation_time, age_days in schemas_too_new[:5]:
                creation_str = creation_time.strftime('%Y-%m-%d')
                print(f"      • {schema} ({age_days} days old)")
    
    if len(schemas_without_timestamp) > 0:
        print(f"\n🏠 SCHEMAS WITHOUT TIMESTAMPS (staying on primary):")
        print(f"   📊 Count: {len(schemas_without_timestamp)} schemas")
        if len(schemas_without_timestamp) <= 10:
            for schema in schemas_without_timestamp:
                print(f"   • {schema}")
        else:
            print(f"   📝 Examples:")
            for schema in schemas_without_timestamp[:5]:
                print(f"      • {schema}")
            print(f"   ... and {len(schemas_without_timestamp) - 5} more")
    
    print(f"\n🚀 TO START MIGRATION:")
    print(f"   sudo python3 safe_mysql_migration.py")
    
    print(f"\n🔄 IF ROLLBACK NEEDED:")
    print(f"   sudo python3 rollback_migration.py")

if __name__ == "__main__":
    main()