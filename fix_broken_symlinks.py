#!/usr/bin/env python3
"""
Fix broken symlinks created during the conversion process.
This script identifies and repairs malformed symlinks in the MySQL directory.
"""

import os
import subprocess
import sys

def get_broken_symlinks():
    """Find all symlinks that might be broken or malformed."""
    mysql_dir = "/app/mysql"
    broken_symlinks = []
    
    try:
        # Get all symlinks
        result = subprocess.run(['sudo', 'find', mysql_dir, '-type', 'l'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            symlinks = result.stdout.strip().split('\n')
            
            for symlink in symlinks:
                if not symlink:
                    continue
                    
                # Check if symlink is broken or malformed
                try:
                    # Try to resolve the symlink
                    resolved = subprocess.run(['sudo', 'readlink', '-f', symlink], 
                                            capture_output=True, text=True)
                    
                    if resolved.returncode != 0:
                        broken_symlinks.append(symlink)
                        continue
                    
                    target = resolved.stdout.strip()
                    
                    # Check if target exists
                    if not os.path.exists(target):
                        broken_symlinks.append(symlink)
                        continue
                    
                    # Check for line breaks in symlink display
                    ls_result = subprocess.run(['sudo', 'ls', '-la', symlink], 
                                             capture_output=True, text=True)
                    
                    if ls_result.returncode == 0 and '\n' in ls_result.stdout:
                        # This indicates a line break in the symlink target
                        if ' -> /' in ls_result.stdout and 'local' in ls_result.stdout.split(' -> /')[1]:
                            broken_symlinks.append(symlink)
                            
                except Exception as e:
                    print(f"Error checking {symlink}: {e}")
                    broken_symlinks.append(symlink)
    
    except Exception as e:
        print(f"Error finding symlinks: {e}")
    
    return broken_symlinks

def fix_symlink(symlink_path):
    """Fix a single broken symlink."""
    try:
        # Extract schema name from path
        schema_name = os.path.basename(symlink_path)
        expected_target = f"/local/mysql_old/{schema_name}"
        
        # Check if target exists
        if not os.path.exists(expected_target):
            print(f"❌ Target does not exist: {expected_target}")
            return False
        
        print(f"🔧 Fixing symlink: {schema_name}")
        
        # Remove broken symlink
        subprocess.run(['sudo', 'rm', symlink_path], check=True)
        
        # Create new symlink
        subprocess.run(['sudo', 'ln', '-s', expected_target, symlink_path], check=True)
        
        # Fix ownership
        subprocess.run(['sudo', 'chown', 'mysql:mysql', symlink_path], check=True)
        
        print(f"✅ Fixed: {schema_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {symlink_path}: {e}")
        return False

def test_mysql_access(schema_name):
    """Test if MySQL can access the schema."""
    try:
        # Try to access the schema
        result = subprocess.run([
            'mysql', '-u', 'root', '-e', 
            f'USE {schema_name}; SHOW TABLES LIMIT 1;'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            if "Tablespace is missing" in result.stderr:
                return "tablespace_error"
            return False
            
    except Exception as e:
        print(f"Error testing MySQL access for {schema_name}: {e}")
        return False

def main():
    print("🔍 FIXING BROKEN SYMLINKS")
    print("=" * 50)
    
    # Find broken symlinks
    broken_symlinks = get_broken_symlinks()
    
    if not broken_symlinks:
        print("✅ No broken symlinks found!")
        return
    
    print(f"📋 Found {len(broken_symlinks)} potentially broken symlinks")
    
    fixed_count = 0
    
    for symlink in broken_symlinks:
        if fix_symlink(symlink):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   🔧 Fixed: {fixed_count}")
    print(f"   ❌ Failed: {len(broken_symlinks) - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n🔄 Restarting MySQL to recognize changes...")
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', 'mysql'], check=True)
            print("✅ MySQL restarted successfully")
        except Exception as e:
            print(f"❌ Error restarting MySQL: {e}")

if __name__ == "__main__":
    main()