#!/usr/bin/env python3
"""
Execute Migration Pipeline
Complete pipeline to execute the schema migration from /dev/nvme0n1p3 to /dev/sda1
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append('/home/admin2/webapp_2')

def run_analysis():
    """Run schema analysis."""
    print("🔍 STEP 1: Schema Analysis")
    print("=" * 50)
    
    from analyze_schema_ages import main as analyze_main
    result = analyze_main()
    
    if result != 0:
        print("❌ Schema analysis failed")
        return False
    
    print("✅ Schema analysis completed")
    return True

def run_setup(skip_if_exists=True):
    """Run /dev/sda1 setup."""
    print("\n🔧 STEP 2: Storage Setup")
    print("=" * 50)
    
    # Check if setup already exists
    sda1_path = Path("/dev/sda1/mysql")
    if skip_if_exists and sda1_path.exists():
        print(f"✅ Setup already exists at {sda1_path}")
        return True
    
    print("⚠️  Storage setup requires manual intervention.")
    print("Please ensure /dev/sda1 is mounted and accessible.")
    print("You may need to run setup_sda1_mysql.py with sudo permissions.")
    
    response = input("Has the /dev/sda1 storage been set up? (y/N): ")
    if response.lower() != 'y':
        print("❌ Storage setup not confirmed")
        return False
    
    print("✅ Storage setup confirmed")
    return True

def run_migration(dry_run=True):
    """Run schema migration."""
    print(f"\n📦 STEP 3: Schema Migration {'(DRY RUN)' if dry_run else '(LIVE)'}")
    print("=" * 50)
    
    from migrate_old_schemas import SchemaMigrator
    
    migrator = SchemaMigrator(dry_run=dry_run)
    success = migrator.run_migration(batch_size=5)  # Smaller batches for better control
    
    if not success:
        print("❌ Schema migration failed")
        return False
    
    print("✅ Schema migration completed")
    return True

def run_webapp_test():
    """Test webapp dual storage access."""
    print("\n🧪 STEP 4: Webapp Integration Test")
    print("=" * 50)
    
    from dual_storage_db_operations import test_dual_storage_access
    
    success = test_dual_storage_access()
    
    if not success:
        print("❌ Webapp integration test failed")
        return False
    
    print("✅ Webapp integration test passed")
    return True

def create_summary_report():
    """Create a summary report of the migration."""
    print("\n📊 STEP 5: Summary Report")
    print("=" * 50)
    
    try:
        from dual_storage_db_operations import get_storage_statistics
        from analyze_schema_ages import format_size
        
        stats = get_storage_statistics()
        
        print("📈 MIGRATION SUMMARY:")
        print("-" * 30)
        
        total_schemas = 0
        total_size = 0
        
        for location, stat in stats['location_stats'].items():
            if stat['count'] > 0:
                size_str = format_size(stat['total_size'])
                print(f"📁 {location.title()}: {stat['count']} schemas, {size_str}")
                total_schemas += stat['count']
                total_size += stat['total_size']
        
        print(f"\n📊 TOTALS:")
        print(f"   Total schemas: {total_schemas}")
        print(f"   Total size: {format_size(total_size)}")
        
        # Storage device information
        storage_info = stats['storage_info']
        
        print(f"\n💾 STORAGE DEVICES:")
        for location, info in storage_info.items():
            if 'free_space' in info and info['free_space'] > 0:
                free_gb = info['free_space'] / (1024**3)
                print(f"   {info['device']}: {free_gb:.2f} GB free")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating summary report: {e}")
        return False

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(description='Execute complete schema migration pipeline')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Perform dry run migration (default)')
    parser.add_argument('--live', action='store_true',
                       help='Perform live migration')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip schema analysis step')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip storage setup step')
    parser.add_argument('--skip-migration', action='store_true',
                       help='Skip migration step (for testing webapp only)')
    parser.add_argument('--skip-webapp-test', action='store_true',
                       help='Skip webapp integration test')
    
    args = parser.parse_args()
    
    # --live overrides --dry-run
    dry_run = not args.live
    
    print("🚀 SCHEMA MIGRATION PIPELINE")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    print("=" * 80)
    
    # Step 1: Analysis
    if not args.skip_analysis:
        if not run_analysis():
            print("❌ Pipeline failed at analysis step")
            return 1
    else:
        print("⏭️  Skipping schema analysis")
    
    # Step 2: Setup
    if not args.skip_setup:
        if not run_setup():
            print("❌ Pipeline failed at setup step")
            return 1
    else:
        print("⏭️  Skipping storage setup")
    
    # Step 3: Migration
    if not args.skip_migration:
        if not run_migration(dry_run=dry_run):
            print("❌ Pipeline failed at migration step")
            return 1
    else:
        print("⏭️  Skipping schema migration")
    
    # Step 4: Webapp test
    if not args.skip_webapp_test:
        if not run_webapp_test():
            print("❌ Pipeline failed at webapp test step")
            return 1
    else:
        print("⏭️  Skipping webapp integration test")
    
    # Step 5: Summary
    create_summary_report()
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    if dry_run:
        print("ℹ️  This was a DRY RUN. To perform live migration, use --live flag")
    else:
        print("✅ Live migration completed successfully!")
        print("\n📋 NEXT STEPS:")
        print("1. Update webapp route handlers to use dual_storage_db_operations")
        print("2. Monitor system performance and disk usage")
        print("3. Set up regular cleanup of old backups")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)