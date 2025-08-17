#!/usr/bin/env python3
"""
Convert ALL migrated schemas to direct SDA1 access
"""

import os
import sys
sys.path.append('/home/admin2/webapp_2')

def convert_all_schemas():
    """Convert all remaining SQL backups to symlink access"""
    
    print("🚀 CONVERTING ALL SCHEMAS TO DIRECT SDA1 ACCESS")
    print("=" * 60)
    
    # Get list of SQL backups
    backup_dir = "/local/mysql_migration_backups"
    sql_files = [f[:-4] for f in os.listdir(backup_dir) if f.endswith('.sql')]
    
    # Check which ones are already converted
    converted_schemas = []
    try:
        import subprocess
        result = subprocess.run(['sudo', 'ls', '-la', '/app/mysql/'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '-> /local/mysql_old/' in line:
                schema_name = line.split()[-3]  # Get symlink name
                converted_schemas.append(schema_name)
    except:
        pass
    
    remaining_schemas = [s for s in sql_files if s not in converted_schemas]
    
    print(f"📊 Status:")
    print(f"   ✅ Already converted: {len(converted_schemas)} schemas")
    print(f"   🔄 Need conversion: {len(remaining_schemas)} schemas")
    print(f"   📁 Total migrated: {len(sql_files)} schemas")
    
    if len(remaining_schemas) == 0:
        print(f"\n🎉 ALL SCHEMAS ALREADY HAVE DIRECT SDA1 ACCESS!")
        return
    
    print(f"\n🎯 Converting remaining {len(remaining_schemas)} schemas...")
    print(f"⏱️  Estimated time: {len(remaining_schemas) * 1.5:.1f} minutes")
    
    # Import the conversion function
    from convert_to_true_dual_storage import convert_sql_backup_to_symlink
    
    success_count = 0
    fail_count = 0
    
    for i, schema in enumerate(remaining_schemas, 1):
        print(f"\n[{i}/{len(remaining_schemas)}] Converting: {schema[:50]}...")
        
        try:
            success = convert_sql_backup_to_symlink(schema, dry_run=False)
            if success:
                success_count += 1
                print(f"   ✅ Success")
            else:
                fail_count += 1
                print(f"   ❌ Failed")
        except Exception as e:
            fail_count += 1
            print(f"   ❌ Error: {e}")
        
        # Progress update every 10 schemas
        if i % 10 == 0:
            print(f"\n📊 Progress: {i}/{len(remaining_schemas)} ({i/len(remaining_schemas)*100:.1f}%)")
            print(f"   ✅ Success: {success_count}")
            print(f"   ❌ Failed: {fail_count}")
    
    print(f"\n" + "="*60)
    print(f"🎉 CONVERSION COMPLETE!")
    print(f"   ✅ Successfully converted: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📊 Total with direct SDA1 access: {len(converted_schemas) + success_count}")
    
    if success_count > 0:
        print(f"\n🚀 ALL CONVERTED SCHEMAS NOW ACCESSIBLE DIRECTLY FROM SDA1!")
        print(f"   • Instant access (no restoration delays)")
        print(f"   • True dual storage achieved")
        print(f"   • Webapp shows blue 'Archive with direct access'")

def convert_batch_schemas(batch_size):
    """Convert a batch of schemas to direct SDA1 access"""
    
    print(f"🚀 CONVERTING {batch_size} SCHEMAS TO DIRECT SDA1 ACCESS")
    print("=" * 60)
    
    # Get list of SQL backups
    backup_dir = "/local/mysql_migration_backups"
    sql_files = [f[:-4] for f in os.listdir(backup_dir) if f.endswith('.sql')]
    
    # Check which ones are already converted
    converted_schemas = []
    try:
        import subprocess
        result = subprocess.run(['sudo', 'ls', '-la', '/app/mysql/'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if '-> /local/mysql_old/' in line:
                schema_name = line.split()[-3]  # Get symlink name
                converted_schemas.append(schema_name)
    except:
        pass
    
    remaining_schemas = [s for s in sql_files if s not in converted_schemas]
    
    if len(remaining_schemas) == 0:
        print(f"🎉 ALL SCHEMAS ALREADY HAVE DIRECT SDA1 ACCESS!")
        return
    
    # Take only the requested batch size
    batch_schemas = remaining_schemas[:batch_size]
    
    print(f"📊 Batch Status:")
    print(f"   ✅ Already converted: {len(converted_schemas)} schemas")
    print(f"   🔄 Converting now: {len(batch_schemas)} schemas")
    print(f"   ⏳ Remaining after this batch: {len(remaining_schemas) - len(batch_schemas)} schemas")
    print(f"   ⏱️  Estimated time: {len(batch_schemas) * 1.5:.1f} minutes")
    
    # Import the conversion function
    from convert_to_true_dual_storage import convert_sql_backup_to_symlink
    
    print(f"\n🎯 Starting batch conversion...")
    success_count = 0
    fail_count = 0
    
    for i, schema in enumerate(batch_schemas, 1):
        print(f"\n[{i}/{len(batch_schemas)}] Converting: {schema[:60]}...")
        
        try:
            success = convert_sql_backup_to_symlink(schema, dry_run=False)
            if success:
                success_count += 1
                print(f"   ✅ Success - Now accessible directly from SDA1!")
            else:
                fail_count += 1
                print(f"   ❌ Failed - Will remain as SQL backup")
        except Exception as e:
            fail_count += 1
            print(f"   ❌ Error: {str(e)[:100]}...")
        
        # Progress update every 10 schemas
        if i % 10 == 0:
            print(f"\n📊 Progress: {i}/{len(batch_schemas)} ({i/len(batch_schemas)*100:.1f}%)")
            print(f"   ✅ Success: {success_count}")
            print(f"   ❌ Failed: {fail_count}")
    
    print(f"\n" + "="*60)
    print(f"🎉 BATCH CONVERSION COMPLETE!")
    print(f"   ✅ Successfully converted: {success_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"   📊 Total with direct SDA1 access: {len(converted_schemas) + success_count}")
    print(f"   🔄 Remaining to convert: {len(remaining_schemas) - len(batch_schemas)}")
    
    if success_count > 0:
        print(f"\n🚀 {success_count} MORE SCHEMAS NOW ACCESSIBLE DIRECTLY FROM SDA1!")
        print(f"   • Test them in your webapp - instant access!")
        print(f"   • Blue 'Archive with direct access' indicator")
    
    if len(remaining_schemas) - len(batch_schemas) > 0:
        print(f"\n🔄 To convert next batch:")
        print(f"   python3 convert_all_to_sda1.py --batch {batch_size}")

def show_conversion_options():
    """Show different conversion options"""
    
    print("🔧 CONVERSION OPTIONS")
    print("=" * 60)
    
    print("1️⃣ **Convert All at Once** (Recommended for overnight)")
    print("   python3 convert_all_to_sda1.py --all")
    print("   • Converts all 1,529 remaining schemas")
    print("   • Takes ~38 hours (1.5 min per schema)")
    print("   • Results in 100% direct SDA1 access")
    
    print("\n2️⃣ **Convert in Batches** (Safer)")
    print("   python3 convert_all_to_sda1.py --batch 50")
    print("   • Converts 50 schemas at a time")
    print("   • Can monitor progress and stop if needed")
    print("   • Gradually increase SDA1 access percentage")
    
    print("\n3️⃣ **Convert Frequently Used** (Smart)")
    print("   • Identify schemas accessed often")
    print("   • Convert those first for immediate benefit")
    print("   • Keep rarely used as SQL backups")
    
    print("\n4️⃣ **Keep Current Mix** (No action needed)")
    print("   • 3 schemas: Direct SDA1 access")
    print("   • 1,529 schemas: Temporary restoration")
    print("   • Convert individual schemas as needed")

if __name__ == "__main__":
    if '--all' in sys.argv:
        convert_all_schemas()
    elif '--batch' in sys.argv:
        try:
            batch_size = int(sys.argv[sys.argv.index('--batch') + 1])
            convert_batch_schemas(batch_size)
        except:
            print("Usage: --batch <number>")
    else:
        show_conversion_options()
        print(f"\n💡 To start conversion:")
        print(f"   python3 convert_all_to_sda1.py --all")