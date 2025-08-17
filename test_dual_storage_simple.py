#!/usr/bin/env python3
"""
Simple Dual Storage Test
Tests the dual storage system using only MySQL queries (no file system access required).
"""

import sys
import mysql.connector
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append('/home/admin2/webapp_2')

def test_mysql_connection():
    """Test basic MySQL connection."""
    try:
        from db_operations import create_connection
        
        print("🔍 Testing MySQL connection...")
        conn = create_connection()
        cursor = conn.cursor()
        
        # Test basic operations
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f"✅ MySQL version: {version}")
        
        # Get datadir
        cursor.execute("SELECT @@datadir")
        datadir = cursor.fetchone()[0]
        print(f"📁 MySQL datadir: {datadir}")
        
        # Count databases
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        db_count = len([db for db in databases if db[0] not in ['information_schema', 'mysql', 'performance_schema', 'sys']])
        print(f"📊 User databases: {db_count}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return False

def test_schema_analysis():
    """Test the schema analysis functions."""
    try:
        print("\n🔍 Testing schema analysis...")
        
        from analyze_schema_ages import extract_timestamp_from_name, get_schema_sizes, format_size
        
        # Test timestamp extraction
        test_schemas = [
            "fuxicai_AGATE_tt164_0_5a8fdb7b_ecc_20250801231213",
            "apierre_FLINT_TT12_5_59e9da29_PFT100US_20250429160953",
            "schema_without_timestamp",
            "maxzhang_FLINT_20240101120000"
        ]
        
        print("📅 Testing timestamp extraction:")
        for schema in test_schemas:
            timestamp = extract_timestamp_from_name(schema)
            if timestamp:
                age_days = (datetime.now() - timestamp).days
                print(f"   {schema[:50]:<50} -> {timestamp.strftime('%Y-%m-%d %H:%M')} ({age_days} days)")
            else:
                print(f"   {schema[:50]:<50} -> No timestamp")
        
        # Test schema size analysis
        print("\n💾 Testing schema size analysis...")
        schema_sizes = get_schema_sizes()
        
        if schema_sizes:
            print(f"✅ Retrieved sizes for {len(schema_sizes)} schemas")
            
            # Show top 5 largest schemas
            sorted_schemas = sorted(schema_sizes.items(), key=lambda x: x[1], reverse=True)
            print("📊 Top 5 largest schemas:")
            for i, (name, size) in enumerate(sorted_schemas[:5]):
                print(f"   {i+1}. {name[:50]:<50} {format_size(size)}")
        else:
            print("❌ No schema sizes retrieved")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Schema analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_migration_categorization():
    """Test schema categorization for migration."""
    try:
        print("\n📂 Testing migration categorization...")
        
        from analyze_schema_ages import extract_timestamp_from_name, get_schema_sizes
        
        cutoff_date = datetime.now() - timedelta(days=60)
        schema_sizes = get_schema_sizes()
        
        old_schemas = 0
        new_schemas = 0
        no_timestamp = 0
        old_size = 0
        new_size = 0
        
        for schema_name, size in schema_sizes.items():
            creation_time = extract_timestamp_from_name(schema_name)
            
            if creation_time is None:
                no_timestamp += 1
            elif creation_time < cutoff_date:
                old_schemas += 1
                old_size += size
            else:
                new_schemas += 1
                new_size += size
        
        from analyze_schema_ages import format_size
        
        print(f"📊 Migration Analysis:")
        print(f"   Old schemas (>60 days): {old_schemas} schemas, {format_size(old_size)}")
        print(f"   New schemas (≤60 days): {new_schemas} schemas, {format_size(new_size)}")
        print(f"   No timestamp: {no_timestamp} schemas")
        print(f"   Total schemas: {len(schema_sizes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration categorization test failed: {e}")
        return False

def test_dual_storage_functions():
    """Test the dual storage management functions."""
    try:
        print("\n🔄 Testing dual storage functions...")
        
        from dual_storage_db_operations import DualStorageManager
        
        # Create manager instance
        dual_storage = DualStorageManager()
        
        # Test storage info
        print("📊 Testing storage info...")
        storage_info = dual_storage.get_storage_info()
        
        for location, info in storage_info.items():
            print(f"   {location}: {info['path']}")
            if 'schema_count' in info:
                print(f"      Schema count: {info.get('schema_count', 'Unknown')}")
        
        # Test MySQL database listing with enhanced info
        print("\n📋 Testing enhanced database listing...")
        from db_operations import create_connection
        
        conn = create_connection()
        cursor = conn.cursor()
        
        enhanced_databases = dual_storage.get_all_databases_enhanced(cursor)
        
        cursor.close()
        conn.close()
        
        if enhanced_databases:
            print(f"✅ Retrieved {len(enhanced_databases)} databases with enhanced info")
            
            # Show first 5 with location info
            print("📂 Sample databases with location info:")
            for i, db_info in enumerate(enhanced_databases[:5]):
                age_str = f"{db_info['age_days']} days" if db_info['age_days'] else "N/A"
                print(f"   {i+1}. {db_info['name'][:40]:<40} {db_info['location']:<12} {age_str}")
        else:
            print("❌ No enhanced database info retrieved")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dual storage functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 DUAL STORAGE SYSTEM - SIMPLE TESTS")
    print("=" * 60)
    print("This test suite validates the system using only MySQL queries")
    print("(No file system access required)")
    print("=" * 60)
    
    tests = [
        ("MySQL Connection", test_mysql_connection),
        ("Schema Analysis", test_schema_analysis),
        ("Migration Categorization", test_migration_categorization),
        ("Dual Storage Functions", test_dual_storage_functions)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("The dual storage system is ready for migration.")
        return 0
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")
        print("Please fix the issues before proceeding with migration.")
        return 1

if __name__ == "__main__":
    exit(main())