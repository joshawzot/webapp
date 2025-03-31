import os
import datetime

# NAS Configuration - Source directory for backups
NAS_BACKUP_DIR = "/databk/dumps"

def main():
    """List all .sql files with '2025' in their names from the backup directory"""
    if not os.path.exists(NAS_BACKUP_DIR):
        print(f"Error: Backup directory {NAS_BACKUP_DIR} does not exist.")
        return
    
    files_with_2025 = []
    
    for filename in os.listdir(NAS_BACKUP_DIR):
        if '2025' in filename and filename.endswith('.sql'):
            files_with_2025.append(filename)
    
    print(f"Found {len(files_with_2025)} files containing '2025' in their names:")
    
    # Print sorted list of files
    for idx, filename in enumerate(sorted(files_with_2025), 1):
        file_path = os.path.join(NAS_BACKUP_DIR, filename)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"{idx}. {filename} ({file_size:.2f} MB)")
    
    print("\nTo restore these schemas, run the restore_2025_schemas.py script.")

if __name__ == '__main__':
    main() 