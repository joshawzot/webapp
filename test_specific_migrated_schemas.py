#!/usr/bin/env python3
"""
Test storage detection with specific migrated schemas from the log
"""

import sys
sys.path.append('/home/admin2/webapp_2')

from dual_storage_db_operations import DualStorageManager
from analyze_schema_ages import extract_timestamp_from_name
from db_operations import create_connection
from pathlib import Path
import json
import subprocess

def test_specific_migrated_schemas():
    """Test storage detection with schemas we know were migrated"""
    print("🧪 TESTING STORAGE DETECTION WITH KNOWN MIGRATED SCHEMAS")
    print("=" * 80)
    
    try:
        # Read migration log to get actually migrated schemas
        migration_log_path = Path('/home/admin2/webapp_2/migration_log_20250803_043553.json')
        migrated_schemas = []
        
        if migration_log_path.exists():
            with open(migration_log_path, 'r') as f:
                migration_log = json.load(f)
            
            # Get schemas that were successfully migrated
            for entry in migration_log:
                if (entry.get('action') == 'VERIFY' and 
                    entry.get('status') == 'success'):
                    migrated_schemas.append(entry.get('schema'))
            
            print(f"📋 Found {len(migrated_schemas)} migrated schemas in log")
        
        # Test with first 10 migrated schemas
        test_schemas = migrated_schemas[:10]
        print(f"🔍 Testing detection with {len(test_schemas)} known migrated schemas:")
        print("-" * 80)
        
        # Create dual storage manager
        dual_storage = DualStorageManager()
        
        # Test each schema
        correct_detections = 0
        for i, schema_name in enumerate(test_schemas, 1):
            print(f"{i:2d}. Testing: {schema_name[:60]:<60}", end=" ")
            
            # Get detection result
            location = dual_storage.get_schema_location(schema_name)
            
            # Check if it's actually in archive directory
            try:
                result = subprocess.run(
                    ['sudo', 'ls', '-d', f'/local/mysql/data/{schema_name}'],
                    capture_output=True, text=True, check=False
                )
                actually_in_archive = result.returncode == 0
            except:
                actually_in_archive = False
            
            # Check correctness
            expected_location = 'secondary' if actually_in_archive else 'primary'
            is_correct = location == expected_location
            
            if is_correct:
                correct_detections += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"→ {status} Detected: {location:<9} (Actually: {'archive' if actually_in_archive else 'primary'})")
        
        print("-" * 80)
        accuracy = (correct_detections / len(test_schemas)) * 100 if test_schemas else 0
        print(f"🎯 ACCURACY: {correct_detections}/{len(test_schemas)} = {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("✅ EXCELLENT: Detection logic is working correctly!")
        elif accuracy >= 70:
            print("⚠️  GOOD: Detection logic is mostly working, minor issues.")
        else:
            print("❌ POOR: Detection logic needs improvement.")
            
            # Debug first failed case
            for i, schema_name in enumerate(test_schemas):
                location = dual_storage.get_schema_location(schema_name)
                try:
                    result = subprocess.run(
                        ['sudo', 'ls', '-d', f'/local/mysql/data/{schema_name}'],
                        capture_output=True, text=True, check=False
                    )
                    actually_in_archive = result.returncode == 0
                except:
                    actually_in_archive = False
                
                if location != ('secondary' if actually_in_archive else 'primary'):
                    print(f"\n🔍 DEBUGGING FAILED CASE: {schema_name}")
                    print(f"   Detected location: {location}")
                    print(f"   Actually in archive: {actually_in_archive}")
                    
                    # Check each detection method
                    print("   Detection method analysis:")
                    
                    # Method 1: sudo ls check
                    try:
                        result = subprocess.run(
                            ['sudo', 'ls', '-d', f'/local/mysql/data/{schema_name}'],
                            capture_output=True, text=True, check=False
                        )
                        print(f"   - sudo ls return code: {result.returncode}")
                    except Exception as e:
                        print(f"   - sudo ls failed: {e}")
                    
                    # Method 2: migration log check
                    in_log = schema_name in migrated_schemas
                    print(f"   - In migration log: {in_log}")
                    
                    # Method 3: age-based check
                    creation_time = extract_timestamp_from_name(schema_name)
                    if creation_time:
                        from datetime import datetime, timedelta
                        age_days = (datetime.now() - creation_time).days
                        is_old = creation_time < (datetime.now() - timedelta(days=60))
                        print(f"   - Age: {age_days} days, Old (>60): {is_old}")
                    else:
                        print(f"   - No timestamp in name")
                    
                    break
        
        return accuracy >= 70
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_specific_migrated_schemas()