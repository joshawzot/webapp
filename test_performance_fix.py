#!/usr/bin/env python3
"""
Test the performance fix for storage detection
Should be fast now since it doesn't update entire cache
"""

import time
import sys
sys.path.append('/home/admin2/webapp_2')
from dual_storage_db_operations import DualStorageManager

def main():
    print("🚀 TESTING PERFORMANCE FIX")
    print("=" * 60)
    
    # Test creating a new instance (should be instant now)
    print("1️⃣ Testing DualStorageManager creation...")
    start_time = time.time()
    dual_storage = DualStorageManager()
    creation_time = time.time() - start_time
    print(f"   ⏱️  Creation time: {creation_time:.2f} seconds")
    
    if creation_time < 1.0:
        print("   ✅ FAST: Creation under 1 second")
    else:
        print("   ❌ SLOW: Creation over 1 second")
    
    # Test individual schema lookup (should be fast)
    print("\n2️⃣ Testing individual schema lookup...")
    test_schema = "MaxZhang_Cullinan_2331_JH1_I03_KPI_20250327082452"
    
    start_time = time.time()
    location = dual_storage.get_schema_location(test_schema)
    lookup_time = time.time() - start_time
    print(f"   📁 Schema location: {location}")
    print(f"   ⏱️  Lookup time: {lookup_time:.2f} seconds")
    
    if lookup_time < 2.0:
        print("   ✅ FAST: Lookup under 2 seconds")
    else:
        print("   ❌ SLOW: Lookup over 2 seconds")
    
    # Test second lookup (should be cached and instant)
    print("\n3️⃣ Testing cached lookup...")
    start_time = time.time()
    location2 = dual_storage.get_schema_location(test_schema)
    cached_time = time.time() - start_time
    print(f"   📁 Schema location: {location2}")
    print(f"   ⏱️  Cached lookup time: {cached_time:.3f} seconds")
    
    if cached_time < 0.1:
        print("   ✅ FAST: Cached lookup under 0.1 seconds")
    else:
        print("   ❌ SLOW: Cached lookup over 0.1 seconds")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    total_time = creation_time + lookup_time + cached_time
    print(f"   🔧 Instance creation: {creation_time:.2f}s")
    print(f"   🔍 First lookup: {lookup_time:.2f}s") 
    print(f"   💾 Cached lookup: {cached_time:.3f}s")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    
    if total_time < 5.0:
        print(f"\n🎉 PERFORMANCE FIX SUCCESS!")
        print(f"   • Total time under 5 seconds")
        print(f"   • Webapp should load much faster now")
        print(f"   • No more long delays when viewing schemas")
    else:
        print(f"\n⚠️  STILL SLOW:")
        print(f"   • Total time: {total_time:.2f} seconds")
        print(f"   • May need additional optimization")

if __name__ == "__main__":
    main()