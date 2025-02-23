import subprocess
from urllib.parse import quote_plus
import shutil
import sys
import re

# Initialize DB_CONFIG
DB_CONFIG = {}

# Local mysql on lenovoi7
DB_CONFIG['RDS_PORT'] = 3306
DB_CONFIG['DB_HOST'] = "localhost"
DB_CONFIG['DB_USER'] = "root"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = ''

#for remote user on other machine
'''DB_CONFIG['RDS_PORT'] = 3306  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "192.168.68.164"
DB_CONFIG['DB_USER'] = "remote_user"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = '' '''

# Common for all configurations
DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])

def check_pv_installed():
    """Check if pv (pipe viewer) is installed."""
    return shutil.which('pv') is not None

def install_pv():
    """Provide instructions to install pv."""
    print("The 'pv' command is not installed. This is needed to show progress.")
    if sys.platform.startswith('linux'):
        print("To install pv on Linux, run:")
        print("sudo apt-get install pv  # For Debian/Ubuntu")
        print("sudo yum install pv      # For CentOS/RHEL")
    elif sys.platform == 'darwin':
        print("To install pv on macOS, run:")
        print("brew install pv")
    return False

def get_schemas_from_dump(input_file):
    """Extract schema names from the dump file."""
    schemas = set()
    try:
        with open(input_file, 'r') as f:
            for line in f:
                # Match CREATE DATABASE or CREATE SCHEMA statements
                create_match = re.search(r'CREATE\s+(DATABASE|SCHEMA)\s+`?([^`\s]+)`?', line, re.IGNORECASE)
                if create_match:
                    schemas.add(create_match.group(2))
                
                # Match database name from dump header
                header_match = re.search(r'-- Host:\s+\S+\s+Database:\s+(\S+)', line)
                if header_match:
                    schemas.add(header_match.group(1))
                
                # Match USE statements
                use_match = re.search(r'USE\s+`?([^`\s]+)`?', line, re.IGNORECASE)
                if use_match:
                    schemas.add(use_match.group(1))

    except Exception as e:
        print(f"Error reading dump file: {e}")
        print(f"Exception details: {str(e)}")
    return schemas

def get_existing_schemas():
    """Get list of existing schemas in MySQL."""
    command = f"mysql -h {DB_CONFIG['DB_HOST']} -u{DB_CONFIG['DB_USER']} -p{DB_CONFIG['MYSQL_PASSWORD']} --port={DB_CONFIG.get('RDS_PORT', 3306)} -N -e 'SHOW DATABASES'"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return set(result.stdout.strip().split('\n'))
    except subprocess.CalledProcessError as e:
        print(f"Error getting existing schemas: {e}")
        return set()

def import_databases(input_file):
    """Import databases from a .sql dump file, skipping existing schemas."""
    try:
        # Check if the file exists
        with open(input_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
        return

    # Get schemas from dump file
    dump_schemas = get_schemas_from_dump(input_file)
    total_schemas = len(dump_schemas)
    
    if not dump_schemas:
        print("No schemas found in dump file or error reading dump file.")
        return

    # Get existing schemas
    existing_schemas = get_existing_schemas()
    
    # Find schemas to skip
    schemas_to_skip = dump_schemas.intersection(existing_schemas)
    schemas_to_restore = dump_schemas - existing_schemas
    skipped_count = len(schemas_to_skip)
    to_restore_count = len(schemas_to_restore)

    if schemas_to_skip:
        print(f"Following schemas already exist and will be skipped: {', '.join(schemas_to_skip)}")
    
    if not schemas_to_restore:
        print("\nRestoration Summary:")
        print(f"Total schemas in dump file: {total_schemas}")
        print(f"Schemas skipped (already exist): {skipped_count}")
        print(f"Schemas attempted to restore: 0")
        print(f"Schemas successfully restored: 0")
        print("All schemas from dump file already exist. Nothing to restore.")
        return

    print(f"Will restore the following schemas: {', '.join(schemas_to_restore)}")

    # Create a temporary file with only the schemas we want to restore
    successfully_restored = set()
    try:
        with open(input_file, 'r') as source, open(f"{input_file}.filtered", 'w') as target:
            current_schema = None
            for line in source:
                # Check for USE or CREATE DATABASE statements
                schema_match = re.search(r'(USE|CREATE\s+DATABASE|CREATE\s+SCHEMA)\s+`?([^`\s]+)`?', line, re.IGNORECASE)
                if schema_match:
                    current_schema = schema_match.group(2)
                
                # Write line only if it belongs to a schema we want to restore
                if current_schema in schemas_to_restore:
                    target.write(line)

        # Import the filtered dump
        if not check_pv_installed():
            if not install_pv():
                print("Continuing without progress indication...")
                command = f"mysql -h {DB_CONFIG['DB_HOST']} -u{DB_CONFIG['DB_USER']} -p{DB_CONFIG['MYSQL_PASSWORD']} --port={DB_CONFIG.get('RDS_PORT', 3306)} < {input_file}.filtered"
            else:
                return
        else:
            # Use pv to show progress while importing
            command = f"pv {input_file}.filtered | mysql -h {DB_CONFIG['DB_HOST']} -u{DB_CONFIG['DB_USER']} -p{DB_CONFIG['MYSQL_PASSWORD']} --port={DB_CONFIG.get('RDS_PORT', 3306)}"
        
        try:
            subprocess.run(command, shell=True, check=True)
            # Check which schemas were actually restored
            new_existing_schemas = get_existing_schemas()
            successfully_restored = schemas_to_restore.intersection(new_existing_schemas)
            print(f"Successfully imported new schemas from {input_file}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while importing the databases: {e}")
        finally:
            # Clean up temporary file
            subprocess.run(f"rm {input_file}.filtered", shell=True)

    except Exception as e:
        print(f"Error during import process: {e}")
    
    # Print final statistics
    print("\nRestoration Summary:")
    print(f"Total schemas in dump file: {total_schemas}")
    print(f"Schemas skipped (already exist): {skipped_count}")
    print(f"Schemas attempted to restore: {to_restore_count}")
    print(f"Schemas successfully restored: {len(successfully_restored)}")
    if successfully_restored:
        print(f"Successfully restored schemas: {', '.join(successfully_restored)}")
    if len(successfully_restored) < to_restore_count:
        failed_schemas = schemas_to_restore - successfully_restored
        print(f"Failed to restore schemas: {', '.join(failed_schemas)}")

def main():
    input_file = input("Enter the path of the .sql dump file to import: ")
    import_databases(input_file)

if __name__ == "__main__":
    main() 