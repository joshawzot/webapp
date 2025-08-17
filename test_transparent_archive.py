#!/usr/bin/env python3
"""
Test transparent archive access functionality
"""

import sys
import os
sys.path.append('/home/admin2/webapp_2')

def test_transparent_archive():
    """Test the transparent archive access system"""
    
    print("🧪 Testing Transparent Archive Access System")
    print("=" * 60)
    
    try:
        from transparent_archive_access import transparent_archive
        
        # Test schema that we know is archived
        schema_name = "MaxZhang_Cullinan_2428_KG2_H09_310_20250417102127"
        
        print(f"📋 Testing schema: {schema_name}")
        print("-" * 60)
        
        # Test 1: Check if schema is archived
        print("1️⃣ Checking if schema is archived...")
        is_archived = transparent_archive.is_schema_archived(schema_name)
        print(f"   Result: {'✅ Yes' if is_archived else '❌ No'}")
        
        # Test 2: Check if schema is currently accessible
        print("\n2️⃣ Checking if schema is currently accessible...")
        is_accessible = transparent_archive.is_schema_accessible(schema_name)
        print(f"   Result: {'✅ Yes' if is_accessible else '❌ No'}")
        
        # Test 3: Get comprehensive schema info
        print("\n3️⃣ Getting comprehensive schema information...")
        schema_info = transparent_archive.get_schema_info(schema_name)
        print(f"   📍 Status: {schema_info['status']}")
        print(f"   📦 Archived: {schema_info['archived']}")
        print(f"   ⚡ Accessible: {schema_info['accessible']}")
        print(f"   🔄 Temp Restored: {schema_info['temp_restored']}")
        print(f"   🎨 Display: {schema_info['storage_display']}")
        print(f"   🏷️ Color: {schema_info['storage_color']}")
        
        # Test 4: Ensure schema accessibility (this will restore if needed)
        print("\n4️⃣ Ensuring schema accessibility...")
        print("   (This will automatically restore from archive if needed)")
        
        success = transparent_archive.ensure_schema_accessible(schema_name)
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            # Test 5: Verify schema is now accessible
            print("\n5️⃣ Verifying schema is now accessible...")
            is_accessible_now = transparent_archive.is_schema_accessible(schema_name)
            print(f"   Result: {'✅ Yes' if is_accessible_now else '❌ No'}")
            
            # Test 6: Get updated schema info
            print("\n6️⃣ Getting updated schema information...")
            updated_info = transparent_archive.get_schema_info(schema_name)
            print(f"   📍 Status: {updated_info['status']}")
            print(f"   🔄 Temp Restored: {updated_info['temp_restored']}")
            print(f"   🎨 Display: {updated_info['storage_display']}")
            
            if updated_info['temp_restored']:
                print("   ✅ Schema successfully restored from archive!")
                print("   📋 The webapp can now access this schema normally")
            
        # Test 7: Check backup file exists
        print("\n7️⃣ Checking backup file...")
        backup_path = transparent_archive.get_backup_file_path(schema_name)
        backup_exists = os.path.exists(backup_path)
        print(f"   Path: {backup_path}")
        print(f"   Exists: {'✅ Yes' if backup_exists else '❌ No'}")
        
        if backup_exists:
            backup_size = os.path.getsize(backup_path) / (1024*1024)
            print(f"   Size: {backup_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def show_webapp_integration():
    """Show how this integrates with the webapp"""
    
    print("\n" + "=" * 60)
    print("🔗 WEBAPP INTEGRATION")
    print("=" * 60)
    
    print("\n🎯 How it works now:")
    print("1. User clicks on archived schema from home page")
    print("2. Webapp calls transparent_archive.ensure_schema_accessible()")
    print("3. System automatically restores schema from backup if needed")
    print("4. Schema becomes accessible in MySQL temporarily")
    print("5. User sees normal table list with orange 'temp restored' indicator")
    print("6. All normal operations work (view tables, download data, etc.)")
    print("7. Schema can be cleaned up later if needed")
    
    print("\n✅ Benefits:")
    print("- No manual restore needed")
    print("- Seamless user experience")
    print("- All webapp features work normally")
    print("- Clear indication of archive status")
    print("- Automatic cleanup available")
    
    print("\n📋 Storage Indicators:")
    print("- 🟢 Green: Primary storage (fast NVMe)")
    print("- 🟠 Orange: Archive temporarily restored")
    print("- 🔵 Blue: Archive (not accessible)")

if __name__ == "__main__":
    test_transparent_archive()
    show_webapp_integration()