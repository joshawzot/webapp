#!/usr/bin/env python3
"""
Quick test for the specific schema the user mentioned
"""

import sys
sys.path.append('/home/admin2/webapp_2')
from dual_storage_db_operations import DualStorageManager

# Create a fresh instance (not the global cached one)
fresh_dual_storage = DualStorageManager()

# Test the specific schema
test_schema = "MaxZhang_Cullinan_2331_JH1_I03_KPI_20250327082452"

print(f"🎯 Testing: {test_schema}")

storage_location = fresh_dual_storage.get_schema_location(test_schema)
print(f"📁 Storage location: {storage_location}")

if storage_location == 'primary':
    print("✅ SUCCESS: Schema correctly shows as primary!")
    print("🔄 Need to restart webapp to pick up the fix")
elif storage_location == 'secondary':
    print("❌ Still shows as secondary - need additional fixes")
else:
    print(f"⚠️  Unknown location: {storage_location}")