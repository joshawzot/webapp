#!/usr/bin/env python3
"""
Quick status check after nuclear fix
"""

import subprocess

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip() if e.stderr else str(e)

def main():
    print("📊 QUICK STATUS CHECK")
    print("=" * 40)
    
    # Check total databases
    success, output = run_cmd(['mysql', '-u', 'root', '-e', 'SHOW DATABASES;'])
    if success:
        databases = output.split('\n')[1:]  # Skip header
        maxzhang_count = len([db for db in databases if 'MaxZhang' in db])
        print(f"🔍 Total MaxZhang schemas visible: {maxzhang_count}")
        
        if maxzhang_count >= 3:
            print("✅ Nuclear fix seems to have worked for test schemas")
            
            # Test one schema
            test_schema = None
            for db in databases:
                if 'MaxZhang_Cullinan_183_100Read' in db:
                    test_schema = db
                    break
            
            if test_schema:
                print(f"🧪 Testing schema: {test_schema}")
                success, output = run_cmd(['mysql', '-u', 'root', '-e', f'USE `{test_schema}`; SHOW TABLES;'])
                if success:
                    table_lines = output.split('\n')[1:] if '\n' in output else []
                    table_count = len([t for t in table_lines if t.strip()])
                    print(f"   📊 Tables found: {table_count}")
                    
                    if table_count > 0:
                        print("   🎉 NUCLEAR FIX WORKED! Tables are visible")
                        print("\n✅ NEXT STEPS:")
                        print("   1. Nuclear fix successful for test schemas")
                        print("   2. Need to apply to remaining 1,479 schemas") 
                        print("   3. Run: sudo python3 nuclear_fix_remaining.py")
                    else:
                        print("   ⚠️  Schemas exist but no tables visible")
                        print("   💡 Need table discovery script")
                else:
                    print(f"   ❌ Cannot access schema: {output}")
        else:
            print("❌ Nuclear fix didn't work - schemas not visible")
            print("💡 Need to retry nuclear fix")
    else:
        print(f"❌ Cannot connect to MySQL: {output}")
    
    # Check backup count
    success, output = run_cmd(['sudo', 'ls', '/local/mysql_corrupted_backup/', '2>/dev/null'])
    if success:
        backup_count = len(output.split('\n')) if output else 0
        print(f"\n📦 Schemas backed up: {backup_count}")
        print(f"📈 Remaining to fix: {1482 - backup_count}")
    
    print(f"\n🎯 RECOMMENDED ACTION:")
    if maxzhang_count >= 3:
        print("✅ Continue with remaining schemas using nuclear fix")
    else:
        print("🔧 Retry nuclear fix for test schemas first")

if __name__ == "__main__":
    main()