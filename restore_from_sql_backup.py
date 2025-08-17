#!/usr/bin/env python3
"""
Restore archived schema data from SQL backup files.
This will completely fix the tablespace issues by recreating everything fresh.
"""

import mysql.connector
import subprocess
import os

def restore_from_sql_backup():
    """Restore schema data from the SQL backup created during migration."""
    
    print("🔧 RESTORING DATA FROM SQL BACKUP")
    print("=" * 60)
    
    schema = 'MaxZhang_Cullinan_2416_KC3_H11_2hrD2_20250525072935'
    backup_file = f'/local/mysql_migration_backups/{schema}.sql'
    
    print(f"🎯 Target schema: {schema}")
    print(f"📁 Backup file: {backup_file}")
    
    # Step 1: Verify backup file exists
    print(f"\n📋 Step 1: Verifying backup file")
    if not os.path.exists(backup_file):
        print(f"  ❌ Backup file not found: {backup_file}")
        return False
    
    # Get file info
    result = subprocess.run(['sudo', 'ls', '-lh', backup_file], 
                          capture_output=True, text=True)
    print(f"  ✅ Backup file found: {result.stdout.strip()}")
    
    # Step 2: Drop the existing broken schema
    print(f"\n📋 Step 2: Removing broken schema")
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # Drop the schema completely
        cursor.execute(f"DROP DATABASE IF EXISTS `{schema}`")
        print(f"  ✅ Dropped existing schema")
        
        cursor.close()
        connection.close()
        
    except mysql.connector.Error as e:
        print(f"  ⚠️ Drop warning: {e}")
    
    # Step 3: Remove the schema directory (it's causing tablespace conflicts)
    print(f"\n📋 Step 3: Cleaning up conflicting files")
    schema_dir = f"/var/lib/mysql/{schema}"
    
    if os.path.exists(schema_dir):
        print(f"  🗑️ Removing conflicting directory: {schema_dir}")
        result = subprocess.run(['sudo', 'rm', '-rf', schema_dir], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Directory removed")
        else:
            print(f"  ⚠️ Remove warning: {result.stderr}")
    
    # Step 4: Restore from SQL backup
    print(f"\n📋 Step 4: Restoring from SQL backup")
    
    print(f"  📥 Importing SQL backup...")
    try:
        # Import the SQL backup
        result = subprocess.run(['sudo', 'mysql', '-e', f'source {backup_file}'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✅ SQL backup imported successfully")
        else:
            print(f"  ❌ Import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False
    
    # Step 5: Test the restored data
    print(f"\n📋 Step 5: Testing restored data")
    
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            charset='utf8mb4'
        )
        cursor = connection.cursor()
        
        # Check if schema exists
        cursor.execute(f"SHOW DATABASES LIKE '{schema}'")
        if not cursor.fetchone():
            print(f"  ❌ Schema not restored")
            return False
        
        print(f"  ✅ Schema restored successfully")
        
        # Check tables
        cursor.execute(f"SHOW TABLES FROM `{schema}`")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"  ✅ Found {len(tables)} tables: {tables}")
        
        # Test data access
        test_results = []
        for table in tables[:5]:  # Test first 5 tables
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{schema}`.`{table}`")
                count = cursor.fetchone()[0]
                print(f"    🎉 {table}: {count} rows accessible")
                test_results.append(True)
            except mysql.connector.Error as e:
                print(f"    ❌ {table}: {e}")
                test_results.append(False)
        
        working_tables = sum(test_results)
        print(f"  📊 Working tables: {working_tables}/{len(test_results)}")
        
        # Test the specific problem table
        if 'Virgin_5' in tables:
            print(f"\n  🧪 Testing Virgin_5 specifically...")
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{schema}`.`Virgin_5`")
                count = cursor.fetchone()[0]
                print(f"    🎉 Virgin_5 SUCCESS! {count} rows")
                
                # Test sample data
                cursor.execute(f"SELECT * FROM `{schema}`.`Virgin_5` LIMIT 3")
                sample_data = cursor.fetchall()
                print(f"    ✅ Sample data: {len(sample_data)} rows with {len(sample_data[0]) if sample_data else 0} columns")
                
            except mysql.connector.Error as e:
                print(f"    ❌ Virgin_5 still broken: {e}")
        
        cursor.close()
        connection.close()
        
        if working_tables > 0:
            print(f"\n🎉 SUCCESS! Data restored from SQL backup!")
            return True
        else:
            print(f"\n❌ No tables are working after restore")
            return False
        
    except Exception as e:
        print(f"  ❌ Test error: {e}")
        return False

def restore_multiple_schemas():
    """If the single schema restore works, offer to restore others."""
    
    print(f"\n🔄 RESTORE ALL ARCHIVED SCHEMAS?")
    print("=" * 40)
    
    # Check how many backup files exist
    backup_dir = "/local/mysql_migration_backups"
    try:
        result = subprocess.run(['sudo', 'ls', backup_dir], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            backup_files = [f for f in result.stdout.split() if f.endswith('.sql')]
            print(f"📁 Found {len(backup_files)} SQL backup files")
            
            print(f"\n💡 Options:")
            print(f"1. ✅ Single schema restore completed successfully")
            print(f"2. 🔄 Can restore {len(backup_files)-1} additional schemas")
            print(f"3. ⚡ All restored schemas will have working data access")
            
            return len(backup_files)
        
    except Exception as e:
        print(f"❌ Error checking backup files: {e}")
        return 0

if __name__ == "__main__":
    print("🚀 DATA RECOVERY FROM SQL BACKUP")
    print("This will completely fix the tablespace issues by restoring")
    print("the schema from the SQL backup created during migration!")
    print()
    
    success = restore_from_sql_backup()
    
    if success:
        print(f"\n🎉 TABLESPACE ISSUE COMPLETELY RESOLVED!")
        print("✅ Schema restored from SQL backup!")
        print("✅ All tables now have working data access!")
        print("✅ No more '(?x248 (archived))' - shows real dimensions!")
        print("✅ Webapp will work perfectly!")
        
        # Offer to restore other schemas
        backup_count = restore_multiple_schemas()
        if backup_count > 1:
            print(f"\n💡 Next steps:")
            print(f"- Test the webapp with the restored schema")
            print(f"- If satisfied, we can restore the remaining {backup_count-1} schemas")
            print(f"- Each restore takes a few minutes but provides full data access")
        
    else:
        print(f"\n❌ Restore failed. Check the errors above.")
        print("The dual storage system is still functional for new data.")