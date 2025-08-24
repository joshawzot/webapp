# db_operations.py
import mysql.connector
from urllib.parse import quote_plus
import os
import json
from datetime import datetime, timedelta

# Initialize DB_CONFIG
DB_CONFIG = {}

# =============================================================================
# IMPORTANT: MySQL Data Storage Location - DUAL STORAGE SETUP
# =============================================================================
# MySQL reports datadir as: /var/lib/mysql/
# However, this is a SYMBOLIC LINK that redirects to: /app/mysql/
# 
# DUAL STORAGE CONFIGURATION:
# - Recent schemas (< 60 days): /app/mysql/ on /dev/nvme2n1p1 (1.8TB disk)
# - Archived schemas (> 60 days): /local/mysql_old/ on /dev/sda1 (9.1TB disk)
# 
# The webapp automatically searches both locations for schemas.
# 
# To verify the actual locations, run:
# $ ls -l /var/lib/mysql
# $ readlink -f /var/lib/mysql  
# $ df -h /app
# $ df -h /local
# =============================================================================

# Local mysql on admin2
DB_CONFIG['RDS_PORT'] = None  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "localhost"
DB_CONFIG['DB_USER'] = "root"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = ''

# ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Aa@2025';
#ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'p@ssw0rd';
# FLUSH PRIVILEGES;
# exit;

#for remote user on other machine
'''DB_CONFIG['RDS_PORT'] = 3306  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "192.168.68.215"
DB_CONFIG['DB_USER'] = "remote_user"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = '' '''

# Common for all configurations
DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])

# =============================================================================
# DUAL STORAGE CONFIGURATION
# =============================================================================
STORAGE_CONFIG = {
    'primary_storage_path': '/app/mysql',
    'archive_storage_path': '/local/mysql_old',
    'primary_storage_mount': '/dev/nvme2n1p1',
    'archive_storage_mount': '/dev/sda1',
    'cutoff_days': 60
}

connection = None

def extract_timestamp_from_schema_name(schema_name):
    """Extract timestamp from schema name if it exists."""
    import re
    patterns = [
        r'_(\d{14})$',  # _20250801231213 at end
        r'_(\d{12})$',  # _202508012312 at end (12 digits)
        r'_(\d{8})$',   # _20250801 at end (8 digits - date only)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, schema_name)
        if match:
            timestamp_str = match.group(1)
            try:
                if len(timestamp_str) == 14:  # YYYYMMDDHHMMSS
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                elif len(timestamp_str) == 12:  # YYYYMMDDHHMM
                    return datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                elif len(timestamp_str) == 8:   # YYYYMMDD
                    return datetime.strptime(timestamp_str, '%Y%m%d')
            except ValueError:
                continue
    return None

def determine_schema_storage_location(schema_name):
    """Determine which storage location a schema should be in based on its age."""
    timestamp = extract_timestamp_from_schema_name(schema_name)
    
    if timestamp is None:
        # No timestamp in name, assume it's in primary storage
        return STORAGE_CONFIG['primary_storage_path']
    
    cutoff_date = datetime.now() - timedelta(days=STORAGE_CONFIG['cutoff_days'])
    
    if timestamp < cutoff_date:
        # Old schema, should be in archive storage
        return STORAGE_CONFIG['archive_storage_path']
    else:
        # Recent schema, should be in primary storage
        return STORAGE_CONFIG['primary_storage_path']

def find_schema_location(schema_name):
    """Find the actual location of a schema by checking both storage locations."""
    # First check if it's a symlink in primary storage (points to archive)
    primary_path = os.path.join(STORAGE_CONFIG['primary_storage_path'], schema_name)
    if os.path.islink(primary_path):
        # This is a symlinked schema - it's physically on archive storage
        return STORAGE_CONFIG['archive_storage_path']
    
    # Check if it exists as a regular directory in primary storage
    if os.path.exists(primary_path):
        return STORAGE_CONFIG['primary_storage_path']
    
    # Check if it exists directly in archive storage
    archive_path = os.path.join(STORAGE_CONFIG['archive_storage_path'], schema_name)
    if os.path.exists(archive_path):
        return STORAGE_CONFIG['archive_storage_path']
    
    # Schema not found in either location
    return None
    return None

def get_mysql_storage_info():
    """Get information about MySQL dual storage configuration."""
    try:
        import subprocess
        
        # Get MySQL's reported datadir
        result = subprocess.run(['mysql', '-u', 'root', '-e', 'SELECT @@datadir;'], 
                              capture_output=True, text=True)
        mysql_datadir = result.stdout.split('\n')[1].strip() if result.returncode == 0 else "Unknown"
        
        # Check if it's a symbolic link
        is_symlink = os.path.islink('/var/lib/mysql')
        actual_path = os.readlink('/var/lib/mysql') if is_symlink else mysql_datadir
        
        # Get primary storage information
        primary_df = subprocess.run(['df', '-h', '/app'], capture_output=True, text=True)
        primary_info = primary_df.stdout.split('\n')[1].split() if primary_df.returncode == 0 else ["Unknown"]
        
        # Get archive storage information
        archive_df = subprocess.run(['df', '-h', '/local'], capture_output=True, text=True)
        archive_info = archive_df.stdout.split('\n')[1].split() if archive_df.returncode == 0 else ["Unknown"]
        
        return {
            'mysql_reported_datadir': mysql_datadir,
            'is_symbolic_link': is_symlink,
            'actual_storage_path': actual_path,
            'dual_storage': True,
            'primary_storage': {
                'path': STORAGE_CONFIG['primary_storage_path'],
                'mount': STORAGE_CONFIG['primary_storage_mount'],
                'disk_device': primary_info[0] if len(primary_info) > 0 else "Unknown",
                'disk_size': primary_info[1] if len(primary_info) > 1 else "Unknown",
                'disk_used': primary_info[2] if len(primary_info) > 2 else "Unknown",
                'disk_available': primary_info[3] if len(primary_info) > 3 else "Unknown"
            },
            'archive_storage': {
                'path': STORAGE_CONFIG['archive_storage_path'],
                'mount': STORAGE_CONFIG['archive_storage_mount'],
                'disk_device': archive_info[0] if len(archive_info) > 0 else "Unknown",
                'disk_size': archive_info[1] if len(archive_info) > 1 else "Unknown",
                'disk_used': archive_info[2] if len(archive_info) > 2 else "Unknown",
                'disk_available': archive_info[3] if len(archive_info) > 3 else "Unknown"
            }
        }
    except Exception as e:
        return {'error': str(e)}



def mysql_name_to_filesystem(schema_name):
    """Convert MySQL database name to filesystem directory name."""
    # MySQL encodes special characters in filesystem directory names
    # Common encodings: - becomes @002d, @ becomes @0040, etc.
    # Note: Order matters! Encode @ first to avoid double-encoding
    encoded_name = schema_name.replace('@', '@0040').replace('-', '@002d')
    return encoded_name

def get_schema_storage_type(schema_name):
    """Get detailed storage type information for a schema including symlink status."""
    import subprocess
    
    # Convert schema name to filesystem encoding
    fs_schema_name = mysql_name_to_filesystem(schema_name)
    
    primary_path = os.path.join(STORAGE_CONFIG['primary_storage_path'], fs_schema_name)
    archive_path = os.path.join(STORAGE_CONFIG['archive_storage_path'], fs_schema_name)
    
    try:
        # Check if it's a symlink in primary storage - use subprocess directly since permissions are restricted
        result = subprocess.run(['sudo', 'test', '-L', primary_path], 
                              capture_output=True, text=True)
        is_symlink = (result.returncode == 0)
        
        if is_symlink:
            # It's a symlink - physically on archive but accessible from primary
            return {
                'type': 'archive_with_direct_access',
                'color': 'info',
                'device': '/dev/sda1 (Archive - Direct Access)',
                'path': archive_path,
                'display': '📦 Archive Storage (Direct Access)',
                'is_symlink': True,
                'access_method': 'symlink'
            }
        
        # Check if it exists as regular directory in primary storage
        result = subprocess.run(['sudo', 'test', '-d', primary_path], 
                              capture_output=True, text=True)
        dir_exists = (result.returncode == 0)
        
        if dir_exists:
            # Regular directory in primary storage
            return {
                'type': 'primary',
                'color': 'success', 
                'device': '/dev/nvme2n1p1 (Primary Drive)',
                'path': primary_path,
                'display': '🟢 Primary Storage',
                'is_symlink': False,
                'access_method': 'direct_primary'
            }
        
        # Check if it exists in archive storage (doesn't need sudo)
        if os.path.exists(archive_path):
            # Only in archive storage (not linked)
            return {
                'type': 'archive_not_linked',
                'color': 'warning',
                'device': '/dev/sda1 (Archive - Not Accessible)',
                'path': archive_path,
                'display': '📦 Archive Storage (Not Linked)',
                'is_symlink': False,
                'access_method': 'none'
            }
        
        # Not found in either location
        return {
            'type': 'not_found',
            'color': 'danger',
            'device': 'Unknown',
            'path': 'Not found',
            'display': '❌ Not Found',
            'is_symlink': False,
            'access_method': 'none'
        }
        
    except Exception as e:
        # Fallback on error
        print(f"Error in storage detection for {schema_name}: {e}")
        return {
            'type': 'not_found',
            'color': 'danger',
            'device': 'Unknown',
            'path': f'Error: {str(e)}',
            'display': '❌ Detection Error',
            'is_symlink': False,
            'access_method': 'none'
        }
def create_connection(database=None):
    """Create a new database connection."""
    try:
        print(f"Creating database connection to: {database}")
        print(f"DB_CONFIG host: {DB_CONFIG.get('DB_HOST', 'NOT SET')}")
        print(f"DB_CONFIG user: {DB_CONFIG.get('DB_USER', 'NOT SET')}")
        print(f"Password set: {'Yes' if DB_CONFIG.get('MYSQL_PASSWORD_RAW') else 'No'}")
        
        # Show dual storage configuration for clarity
        storage_info = get_mysql_storage_info()
        if 'error' not in storage_info and storage_info.get('dual_storage'):
            print(f"MySQL dual storage: Primary {storage_info['primary_storage']['disk_device']}, Archive {storage_info['archive_storage']['disk_device']}")
        elif 'error' not in storage_info:
            print(f"MySQL data stored on: {storage_info['primary_storage']['disk_device']} at {storage_info['actual_storage_path']}")
        
        connection = mysql.connector.connect(
            host=DB_CONFIG['DB_HOST'],
            user=DB_CONFIG['DB_USER'],
            password=DB_CONFIG['MYSQL_PASSWORD_RAW'],
            database=database
        )
        print(f"Successfully connected to database: {database}")
        return connection
    except Exception as conn_error:
        print(f"Database connection failed: {conn_error}")
        import traceback
        traceback.print_exc()
        raise

def create_db(db_name):
    """
    Creates a database with the given name.
    
    Returns:
        True if the database was created successfully or already exists.
        False if there was an error creating the database.
    """
    connection = create_connection()
    if connection is None:
        return False

    cursor = connection.cursor()
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
        print(f"Database '{db_name}' created successfully.")
        return True
    except mysql.connector.Error as err:
        print(f"Failed creating database '{db_name}': {err}")
        return False
    finally:
        cursor.close()
        close_connection()
        
def fetch_data(cursor, query):
    cursor.execute(query)
    return cursor.fetchall()

def close_connection():
    global connection
    if connection is not None and connection.is_connected():
        connection.close()
        connection = None

from sqlalchemy import create_engine

def create_db_engine(db_name):
    engine_url = f"mysql+mysqlconnector://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD']}@{DB_CONFIG['DB_HOST']}/{db_name}"
    engine = create_engine(
        engine_url,
        pool_size=10,  # Maximum number of connections to keep in the pool
        max_overflow=5  # Allow up to 5 additional connections beyond pool_size
    )
    return engine

def get_all_databases(cursor):
    """Fetch all database names from the MySQL server and return them as a list, excluding restricted databases."""
    # Define your restricted patterns or names
    restricted_patterns = ['performance_schema', 'mysql', 'information_schema', 'sys']
    
    try:
        cursor.execute("SHOW DATABASES")
        # Use list comprehension to extract database names from the cursor
        all_databases = [db[0] for db in cursor]

        # Filter your databases list to exclude restricted databases
        filtered_databases = [db for db in all_databases if db not in restricted_patterns]

        return filtered_databases

    except mysql.connector.Error as err:
        print(f"Failed to list databases: {err}")
        # Decide how to handle the error. Here we're returning an empty list, but you might want to re-raise the error or handle it differently.
        return []

def connect_to_db(user, password, host, port=None):
    """Connect to the MySQL server and return the connection."""
    connection_params = {
        'host': host,
        'user': user,
        'password': password
    }
    if port:
        connection_params['port'] = port
    try:
        return mysql.connector.connect(**connection_params)
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        return None  # Return None if there's a connection error

def fetch_tables(database):
    """Fetch table names, creation times, and dimensions from the database."""
    connection = create_connection(database)
    cursor = connection.cursor()
    print("database:", database)

    # Query to fetch table names and creation times
    table_query = """
    SELECT TABLE_NAME, CREATE_TIME
    FROM information_schema.tables
    WHERE table_schema = %s
    AND LEFT(TABLE_NAME, 1) != '_'
    ORDER BY CREATE_TIME DESC;
    """
    cursor.execute(table_query, (database,))
    table_info = cursor.fetchall()

    # Dictionary to store table information with dimensions
    tables = []
    for name, time in table_info:
        try:
            # Count the number of columns
            column_query = """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s;
            """
            cursor.execute(column_query, (database, name))
            column_count = cursor.fetchone()[0]

            # Count the number of rows (this may fail for tablespace issues)
            try:
                row_query = f"SELECT COUNT(*) FROM `{name}`;"
                cursor.execute(row_query)
                row_count = cursor.fetchone()[0]
                
                # Store table info with actual dimensions
                tables.append({'table_name': name, 'creation_time': time, 'dimensions': f"{row_count}x{column_count}"})
            except Exception as row_error:
                # Row count failed (likely tablespace issue), but we still have column count
                if "Tablespace is missing" in str(row_error) or "1812" in str(row_error):
                    # Tablespace issue - show columns but indicate data unavailable
                    tables.append({'table_name': name, 'creation_time': time, 'dimensions': f"?x{column_count} (archived)"})
                else:
                    # Other error
                    print(f"Row count error for table '{name}': {row_error}")
                    tables.append({'table_name': name, 'creation_time': time, 'dimensions': f"?x{column_count} (error)"})
                    
        except Exception as e:
            print(f"Error processing table/view '{name}': {e}")
            # Complete failure - can't even get column count
            tables.append({'table_name': name, 'creation_time': time, 'dimensions': 'ERROR: Invalid view or table'})

    cursor.close()
    connection.close()
    return tables

def rename_database(old_name, new_name):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Create new database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{new_name}`;")

        # Fetch all tables from the old database
        cursor.execute(f"SHOW TABLES FROM `{old_name}`;")
        tables = cursor.fetchall()

        # Move each table to the new database
        for (table_name,) in tables:
            cursor.execute(f"RENAME TABLE `{old_name}`.`{table_name}` TO `{new_name}`.`{table_name}`;")

        # Drop old database
        cursor.execute(f"DROP DATABASE `{old_name}`;")

        # Commit the changes
        connection.commit()
        return True

    except mysql.connector.Error as err:
        print(f"Error while renaming database: {err}")
        connection.rollback()  # Rollback in case of any error
        return False
    finally:
        cursor.close()
        connection.close()

def get_table_from_database(database_name, table_name):
    connection = create_connection(database_name)
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT * FROM `{table_name}`")
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return data, columns
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

# Assume you have a function to get a connection from a pool or create a new one if the pool is empty
def get_db_connection(database=None):
    try:
        connection = mysql.connector.connect(
            pool_name="mypool",
            host=DB_CONFIG['DB_HOST'],
            user=DB_CONFIG['DB_USER'],
            password=DB_CONFIG['MYSQL_PASSWORD'],  # Assuming password is already appropriately handled
            database=database
        )
        if database:
            cursor = connection.cursor()
            cursor.execute(f"USE {database};")
            cursor.close()
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

import csv
import io
import numpy as np

def get_npy_from_table(database, table_name):
    connection = create_connection(database)
    if connection is None:
        raise Exception("Failed to connect to the database.")

    cursor = connection.cursor()
    try:
        query = f"SELECT * FROM `{table_name}`;"
        cursor.execute(query)
        rows = cursor.fetchall()

        if rows:
            array_data = np.array(rows)
            bio = io.BytesIO()
            np.save(bio, array_data, allow_pickle=False)
            bio.seek(0)
            return bio.getvalue()
        else:
            return None  # Consider whether to raise an exception or handle differently
    except Exception as e:
        print(f"Error fetching data from table {table_name}: {e}")
        raise  # Re-raise the exception to be caught by Flask route
    finally:
        cursor.close()
        connection.close()

def get_csv_from_table(database, table_name):
    connection = create_connection(database)
    if connection is None:
        raise Exception("Failed to connect to the database.")

    cursor = connection.cursor()
    try:
        # Construct the SQL query to fetch all data from the specified table
        query = f"SELECT * FROM `{table_name}`;"
        cursor.execute(query)

        # Use StringIO to capture CSV output
        output = io.StringIO()
        csv_writer = csv.writer(output)

        # Write header (column names)
        column_headers = [i[0] for i in cursor.description]
        csv_writer.writerow(column_headers)

        # Write data rows
        for row in cursor.fetchall():
            csv_writer.writerow(row)

        # Get CSV string from StringIO
        csv_string = output.getvalue()
        output.close()

        return csv_string
    except Exception as e:
        print(f"Error fetching data from table {table_name}: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_metadata_csv_from_table(database, table_name):
    """
    Convert bitmap table data to metadata format with BL, WL, value columns.
    BL = row coordinate (Bit Line)
    WL = column coordinate (Word Line)
    value = the actual value at that position
    """
    connection = create_connection(database)
    if connection is None:
        raise Exception("Failed to connect to the database.")

    cursor = connection.cursor()
    try:
        # Construct the SQL query to fetch all data from the specified table
        query = f"SELECT * FROM `{table_name}`;"
        cursor.execute(query)

        # Get column headers
        column_headers = [i[0] for i in cursor.description]
        rows = cursor.fetchall()

        # Use StringIO to capture CSV output
        output = io.StringIO()
        csv_writer = csv.writer(output)

        # Write header for metadata format
        csv_writer.writerow(['BL', 'WL', 'value'])

        # Convert bitmap data to BL, WL, value format
        for bl_index, row in enumerate(rows):
            for wl_index, value in enumerate(row):
                # Skip if value is None or empty
                if value is not None and str(value).strip() != '':
                    csv_writer.writerow([bl_index, wl_index, value])

        # Get CSV string from StringIO
        csv_string = output.getvalue()
        output.close()

        return csv_string
    except Exception as e:
        print(f"Error fetching metadata from table {table_name}: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_pattern_files():
    """
    Get the pattern files dictionary with paths to state pattern files.
    Returns dictionary with appropriate paths (absolute or relative)
    """
    try:
        # Import Flask app to access configuration
        from run import app
        use_absolute_paths = app.config.get('USE_ABSOLUTE_STATE_PATTERN_PATHS', True)
    except:
        # Fallback to True if import fails or config not found
        use_absolute_paths = True
    
    if use_absolute_paths:
        # Absolute paths dictionary
        return {
            "1296x64_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_rowbar_4states.npy",
            "2048x32_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/2048x32_rowbar_4states.npy",
            "2048x32_random": "/home/admin2/webapp_2/State_pattern_files/2048x32_random.npy",
            "3x4_4states_debug": "/home/admin2/webapp_2/State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "/home/admin2/webapp_2/State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "/home/admin2/webapp_2/State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "/home/admin2/webapp_2/State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "/home/admin2/webapp_2/State_pattern_files/248x248_16states.npy",
            "248x248_2states": "/home/admin2/webapp_2/State_pattern_files/248x248_2states.npy",
            "248x248_64states": "/home/admin2/webapp_2/State_pattern_files/248x248_64states.npy",
            "62x62_2states": "/home/admin2/webapp_2/State_pattern_files/62x62_2states.npy",
            "248x1_1state": "/home/admin2/webapp_2/State_pattern_files/248x1_1state.npy",
            "248x256_1state": "/home/admin2/webapp_2/State_pattern_files/248x256_1state.npy",
            "82944x78_ecc_fuxi": "/home/admin2/webapp_2/State_pattern_files/82944x78_ecc_fuxi.npy",
            "65536x78_ecc": "/home/admin2/webapp_2/State_pattern_files/65536x78_ecc.npy",
            "256x32_pr0": "/home/admin2/webapp_2/State_pattern_files/256x32_pr0.npy",
            "256x32_pr1": "/home/admin2/webapp_2/State_pattern_files/256x32_pr1.npy",
            "test_chin": "/home/admin2/webapp_2/State_pattern_files/ecc_new.npy",
            # ECC 2048x32 IO files
            "ecc_2048x32_IO0": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO0.npy",
            "ecc_2048x32_IO1": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO1.npy",
            "ecc_2048x32_IO2": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO2.npy",
            "ecc_2048x32_IO3": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO3.npy",
            "ecc_2048x32_IO4": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO4.npy",
            "ecc_2048x32_IO5": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO5.npy",
            "ecc_2048x32_IO6": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO6.npy",
            "ecc_2048x32_IO7": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO7.npy",
            "ecc_2048x32_IO8": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO8.npy",
            "ecc_2048x32_IO9": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO9.npy",
            "ecc_2048x32_IO10": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO10.npy",
            "ecc_2048x32_IO11": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO11.npy",
            "ecc_2048x32_IO12": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO12.npy",
            "ecc_2048x32_IO13": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO13.npy",
            "ecc_2048x32_IO14": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO14.npy",
            "ecc_2048x32_IO15": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO15.npy",
            "ecc_2048x32_IO16": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO16.npy",
            "ecc_2048x32_IO17": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO17.npy",
            "ecc_2048x32_IO18": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO18.npy",
            "ecc_2048x32_IO19": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO19.npy",
            "ecc_2048x32_IO20": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO20.npy",
            "ecc_2048x32_IO21": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO21.npy",
            "ecc_2048x32_IO22": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO22.npy",
            "ecc_2048x32_IO23": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO23.npy",
            "ecc_2048x32_IO24": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO24.npy",
            "ecc_2048x32_IO25": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO25.npy",
            "ecc_2048x32_IO26": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO26.npy",
            "ecc_2048x32_IO27": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO27.npy",
            "ecc_2048x32_IO28": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO28.npy",
            "ecc_2048x32_IO29": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO29.npy",
            "ecc_2048x32_IO30": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO30.npy",
            "ecc_2048x32_IO31": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO31.npy",
            "ecc_2048x32_IO32": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO32.npy",
            "ecc_2048x32_IO33": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO33.npy",
            "ecc_2048x32_IO34": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO34.npy",
            "ecc_2048x32_IO35": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO35.npy",
            "ecc_2048x32_IO36": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO36.npy",
            "ecc_2048x32_IO37": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO37.npy",
            "ecc_2048x32_IO38": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO38.npy",
            "ecc_2048x32_IO39": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO39.npy",
            "ecc_2048x32_IO40": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO40.npy",
            "ecc_2048x32_IO41": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO41.npy",
            "ecc_2048x32_IO42": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO42.npy",
            "ecc_2048x32_IO43": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO43.npy",
            "ecc_2048x32_IO44": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO44.npy",
            "ecc_2048x32_IO45": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO45.npy",
            "ecc_2048x32_IO46": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO46.npy",
            "ecc_2048x32_IO47": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO47.npy",
            "ecc_2048x32_IO48": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO48.npy",
            "ecc_2048x32_IO49": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO49.npy",
            "ecc_2048x32_IO50": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO50.npy",
            "ecc_2048x32_IO51": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO51.npy",
            "ecc_2048x32_IO52": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO52.npy",
            "ecc_2048x32_IO53": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO53.npy",
            "ecc_2048x32_IO54": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO54.npy",
            "ecc_2048x32_IO55": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO55.npy",
            "ecc_2048x32_IO56": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO56.npy",
            "ecc_2048x32_IO57": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO57.npy",
            "ecc_2048x32_IO58": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO58.npy",
            "ecc_2048x32_IO59": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO59.npy",
            "ecc_2048x32_IO60": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO60.npy",
            "ecc_2048x32_IO61": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO61.npy",
            "ecc_2048x32_IO62": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO62.npy",
            "ecc_2048x32_IO63": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO63.npy",
            "ecc_2048x32_IO64": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO64.npy",
            "ecc_2048x32_IO65": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO65.npy",
            "ecc_2048x32_IO66": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO66.npy",
            "ecc_2048x32_IO67": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO67.npy",
            "ecc_2048x32_IO68": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO68.npy",
            "ecc_2048x32_IO69": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO69.npy",
            "ecc_2048x32_IO70": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO70.npy",
            "ecc_2048x32_IO71": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO71.npy",
            "ecc_2048x32_IO72": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO72.npy",
            "ecc_2048x32_IO73": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO73.npy",
            "ecc_2048x32_IO74": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO74.npy",
            "ecc_2048x32_IO75": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO75.npy",
            "ecc_2048x32_IO76": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO76.npy",
            "ecc_2048x32_IO77": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO77.npy",
        }
    else:
        # Relative paths dictionary
        return {
            "1296x64_rowbar_4states": "State_pattern_files/1296x64_rowbar_4states.npy",
            "2048x32_rowbar_4states": "State_pattern_files/2048x32_rowbar_4states.npy",
            "2048x32_random": "State_pattern_files/2048x32_random.npy",
            "3x4_4states_debug": "State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "State_pattern_files/248x248_16states.npy",
            "248x248_2states": "State_pattern_files/248x248_2states.npy",
            "248x248_64states": "State_pattern_files/248x248_64states.npy",
            "62x62_2states": "State_pattern_files/62x62_2states.npy",
            "248x1_1state": "State_pattern_files/248x1_1state.npy",
            "248x256_1state": "State_pattern_files/248x256_1state.npy",
            "82944x78_ecc_fuxi": "State_pattern_files/82944x78_ecc_fuxi.npy",
            "65536x78_ecc": "State_pattern_files/65536x78_ecc.npy",
            "256x32_pr0": "State_pattern_files/256x32_pr0.npy",
            "256x32_pr1": "State_pattern_files/256x32_pr1.npy",
            "test_chin": "State_pattern_files/ecc_new.npy",
            # ECC 2048x32 IO files
            "ecc_2048x32_IO0": "State_pattern_files/ecc_2048x32_IO0.npy",
            "ecc_2048x32_IO1": "State_pattern_files/ecc_2048x32_IO1.npy",
            "ecc_2048x32_IO2": "State_pattern_files/ecc_2048x32_IO2.npy",
            "ecc_2048x32_IO3": "State_pattern_files/ecc_2048x32_IO3.npy",
            "ecc_2048x32_IO4": "State_pattern_files/ecc_2048x32_IO4.npy",
            "ecc_2048x32_IO5": "State_pattern_files/ecc_2048x32_IO5.npy",
            "ecc_2048x32_IO6": "State_pattern_files/ecc_2048x32_IO6.npy",
            "ecc_2048x32_IO7": "State_pattern_files/ecc_2048x32_IO7.npy",
            "ecc_2048x32_IO8": "State_pattern_files/ecc_2048x32_IO8.npy",
            "ecc_2048x32_IO9": "State_pattern_files/ecc_2048x32_IO9.npy",
            "ecc_2048x32_IO10": "State_pattern_files/ecc_2048x32_IO10.npy",
            "ecc_2048x32_IO11": "State_pattern_files/ecc_2048x32_IO11.npy",
            "ecc_2048x32_IO12": "State_pattern_files/ecc_2048x32_IO12.npy",
            "ecc_2048x32_IO13": "State_pattern_files/ecc_2048x32_IO13.npy",
            "ecc_2048x32_IO14": "State_pattern_files/ecc_2048x32_IO14.npy",
            "ecc_2048x32_IO15": "State_pattern_files/ecc_2048x32_IO15.npy",
            "ecc_2048x32_IO16": "State_pattern_files/ecc_2048x32_IO16.npy",
            "ecc_2048x32_IO17": "State_pattern_files/ecc_2048x32_IO17.npy",
            "ecc_2048x32_IO18": "State_pattern_files/ecc_2048x32_IO18.npy",
            "ecc_2048x32_IO19": "State_pattern_files/ecc_2048x32_IO19.npy",
            "ecc_2048x32_IO20": "State_pattern_files/ecc_2048x32_IO20.npy",
            "ecc_2048x32_IO21": "State_pattern_files/ecc_2048x32_IO21.npy",
            "ecc_2048x32_IO22": "State_pattern_files/ecc_2048x32_IO22.npy",
            "ecc_2048x32_IO23": "State_pattern_files/ecc_2048x32_IO23.npy",
            "ecc_2048x32_IO24": "State_pattern_files/ecc_2048x32_IO24.npy",
            "ecc_2048x32_IO25": "State_pattern_files/ecc_2048x32_IO25.npy",
            "ecc_2048x32_IO26": "State_pattern_files/ecc_2048x32_IO26.npy",
            "ecc_2048x32_IO27": "State_pattern_files/ecc_2048x32_IO27.npy",
            "ecc_2048x32_IO28": "State_pattern_files/ecc_2048x32_IO28.npy",
            "ecc_2048x32_IO29": "State_pattern_files/ecc_2048x32_IO29.npy",
            "ecc_2048x32_IO30": "State_pattern_files/ecc_2048x32_IO30.npy",
            "ecc_2048x32_IO31": "State_pattern_files/ecc_2048x32_IO31.npy",
            "ecc_2048x32_IO32": "State_pattern_files/ecc_2048x32_IO32.npy",
            "ecc_2048x32_IO33": "State_pattern_files/ecc_2048x32_IO33.npy",
            "ecc_2048x32_IO34": "State_pattern_files/ecc_2048x32_IO34.npy",
            "ecc_2048x32_IO35": "State_pattern_files/ecc_2048x32_IO35.npy",
            "ecc_2048x32_IO36": "State_pattern_files/ecc_2048x32_IO36.npy",
            "ecc_2048x32_IO37": "State_pattern_files/ecc_2048x32_IO37.npy",
            "ecc_2048x32_IO38": "State_pattern_files/ecc_2048x32_IO38.npy",
            "ecc_2048x32_IO39": "State_pattern_files/ecc_2048x32_IO39.npy",
            "ecc_2048x32_IO40": "State_pattern_files/ecc_2048x32_IO40.npy",
            "ecc_2048x32_IO41": "State_pattern_files/ecc_2048x32_IO41.npy",
            "ecc_2048x32_IO42": "State_pattern_files/ecc_2048x32_IO42.npy",
            "ecc_2048x32_IO43": "State_pattern_files/ecc_2048x32_IO43.npy",
            "ecc_2048x32_IO44": "State_pattern_files/ecc_2048x32_IO44.npy",
            "ecc_2048x32_IO45": "State_pattern_files/ecc_2048x32_IO45.npy",
            "ecc_2048x32_IO46": "State_pattern_files/ecc_2048x32_IO46.npy",
            "ecc_2048x32_IO47": "State_pattern_files/ecc_2048x32_IO47.npy",
            "ecc_2048x32_IO48": "State_pattern_files/ecc_2048x32_IO48.npy",
            "ecc_2048x32_IO49": "State_pattern_files/ecc_2048x32_IO49.npy",
            "ecc_2048x32_IO50": "State_pattern_files/ecc_2048x32_IO50.npy",
            "ecc_2048x32_IO51": "State_pattern_files/ecc_2048x32_IO51.npy",
            "ecc_2048x32_IO52": "State_pattern_files/ecc_2048x32_IO52.npy",
            "ecc_2048x32_IO53": "State_pattern_files/ecc_2048x32_IO53.npy",
            "ecc_2048x32_IO54": "State_pattern_files/ecc_2048x32_IO54.npy",
            "ecc_2048x32_IO55": "State_pattern_files/ecc_2048x32_IO55.npy",
            "ecc_2048x32_IO56": "State_pattern_files/ecc_2048x32_IO56.npy",
            "ecc_2048x32_IO57": "State_pattern_files/ecc_2048x32_IO57.npy",
            "ecc_2048x32_IO58": "State_pattern_files/ecc_2048x32_IO58.npy",
            "ecc_2048x32_IO59": "State_pattern_files/ecc_2048x32_IO59.npy",
            "ecc_2048x32_IO60": "State_pattern_files/ecc_2048x32_IO60.npy",
            "ecc_2048x32_IO61": "State_pattern_files/ecc_2048x32_IO61.npy",
            "ecc_2048x32_IO62": "State_pattern_files/ecc_2048x32_IO62.npy",
            "ecc_2048x32_IO63": "State_pattern_files/ecc_2048x32_IO63.npy",
            "ecc_2048x32_IO64": "State_pattern_files/ecc_2048x32_IO64.npy",
            "ecc_2048x32_IO65": "State_pattern_files/ecc_2048x32_IO65.npy",
            "ecc_2048x32_IO66": "State_pattern_files/ecc_2048x32_IO66.npy",
            "ecc_2048x32_IO67": "State_pattern_files/ecc_2048x32_IO67.npy",
            "ecc_2048x32_IO68": "State_pattern_files/ecc_2048x32_IO68.npy",
            "ecc_2048x32_IO69": "State_pattern_files/ecc_2048x32_IO69.npy",
            "ecc_2048x32_IO70": "State_pattern_files/ecc_2048x32_IO70.npy",
            "ecc_2048x32_IO71": "State_pattern_files/ecc_2048x32_IO71.npy",
            "ecc_2048x32_IO72": "State_pattern_files/ecc_2048x32_IO72.npy",
            "ecc_2048x32_IO73": "State_pattern_files/ecc_2048x32_IO73.npy",
            "ecc_2048x32_IO74": "State_pattern_files/ecc_2048x32_IO74.npy",
            "ecc_2048x32_IO75": "State_pattern_files/ecc_2048x32_IO75.npy",
            "ecc_2048x32_IO76": "State_pattern_files/ecc_2048x32_IO76.npy",
            "ecc_2048x32_IO77": "State_pattern_files/ecc_2048x32_IO77.npy",
        }

def get_table_dimensions(database, table_name):
    """
    Get the dimensions of a table by examining its structure.
    Returns a tuple (rows, columns) representing the table dimensions.
    """
    connection = create_connection(database)
    if connection is None:
        raise Exception("Failed to connect to the database.")

    cursor = connection.cursor()
    try:
        # Get table column count
        query = f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s"
        cursor.execute(query, (database, table_name))
        column_count = cursor.fetchone()[0]
        
        # Get table row count
        query = f"SELECT COUNT(*) FROM `{table_name}`"
        cursor.execute(query)
        row_count = cursor.fetchone()[0]
        
        return (row_count, column_count)
        
    except Exception as e:
        raise Exception(f"Error getting dimensions for table {table_name}: {str(e)}")
    finally:
        connection.close()

def get_metadata_csv_with_pattern_from_table(database, table_name, state_pattern):
    """
    Convert bitmap table data to metadata format with BL, WL, value, Level columns.
    BL = row coordinate (Bit Line)
    WL = column coordinate (Word Line)
    value = the actual value at that position
    Level = the state/level from the state pattern file
    
    Optimized version using vectorized numpy operations for large datasets.
    """
    import numpy as np
    
    connection = create_connection(database)
    if connection is None:
        raise Exception("Failed to connect to the database.")

    cursor = connection.cursor()
    try:
        print(f"Loading data for table {table_name}...")
        # Construct the SQL query to fetch all data from the specified table
        query = f"SELECT * FROM `{table_name}`;"
        cursor.execute(query)

        # Get column headers
        column_headers = [i[0] for i in cursor.description]
        rows = cursor.fetchall()

        # Convert to numpy array
        print(f"Converting {len(rows)} rows to numpy array...")
        data_matrix = np.array(rows, dtype=float)
        print(f"Data matrix shape: {data_matrix.shape}")
        
        # Load the state pattern file
        print(f"Loading state pattern: {state_pattern}")
        pattern_files = get_pattern_files()
        pattern_file_path = pattern_files.get(state_pattern)
        
        if not pattern_file_path:
            raise Exception(f"State pattern '{state_pattern}' not found")
            
        try:
            pattern_file_array = np.load(pattern_file_path)
            
            # Special handling for specific patterns that need reshaping
            if state_pattern == "82944x78_ecc_fuxi":
                # Reshape the 3D array to 2D (78, 82944) and then transpose to (82944, 78)
                pattern_file_array = pattern_file_array.reshape(78, 82944).T
            elif state_pattern == "65536x78_ecc":
                # Reshape the 3D array to 2D (78, 65536) and then transpose to (65536, 78)
                pattern_file_array = pattern_file_array.reshape(78, 65536).T
            elif state_pattern.startswith("ecc_2048x32_IO"):
                # Transpose from (32, 2048) to (2048, 32) to match expected dimensions
                if len(pattern_file_array.shape) == 2 and pattern_file_array.shape == (32, 2048):
                    pattern_file_array = pattern_file_array.T
                
        except Exception as e:
            raise Exception(f"Error loading state pattern file '{pattern_file_path}': {str(e)}")

        # Ensure that pattern_file_array has the same shape as data_matrix
        if pattern_file_array.shape != data_matrix.shape:
            raise Exception(f"Pattern file shape {pattern_file_array.shape} does not match data matrix shape {data_matrix.shape}")

        print("Creating coordinate meshgrid...")
        # Create coordinate arrays using vectorized operations
        rows, cols = data_matrix.shape
        bl_coords, wl_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        
        print("Preparing all data points for metadata export...")
        # For metadata export, include ALL coordinates (including zeros and NaN values)
        # Users want to see the complete mapping of BL, WL coordinates to their Level values
        
        # Flatten all arrays to 1D for CSV output
        all_bl = bl_coords.flatten()
        all_wl = wl_coords.flatten()
        all_values = data_matrix.flatten()
        all_levels = pattern_file_array.flatten().astype(int)
        
        # Replace NaN values with a placeholder for better CSV readability
        all_values = np.where(np.isnan(all_values), 'NaN', all_values)
        
        print(f"Including all {len(all_values)} coordinate points (complete {rows}x{cols} matrix)")
        
        # Use StringIO to capture CSV output
        output = io.StringIO()
        csv_writer = csv.writer(output)

        # Write header for metadata format with Level column
        csv_writer.writerow(['BL', 'WL', 'value', 'Level'])

        print("Writing CSV data...")
        # Convert to list of rows and write all at once (much faster than individual writerow calls)
        if len(all_values) > 0:
            # Create the data array
            csv_data = np.column_stack((all_bl, all_wl, all_values, all_levels))
            
            # Write in chunks to avoid memory issues with very large datasets
            chunk_size = 50000  # Process 50k rows at a time
            for i in range(0, len(csv_data), chunk_size):
                chunk = csv_data[i:i + chunk_size]
                csv_writer.writerows(chunk.tolist())
                if i % (chunk_size * 10) == 0:  # Progress update every 500k rows
                    print(f"Processed {min(i + chunk_size, len(csv_data))} / {len(csv_data)} rows...")

        print("CSV generation complete!")
        # Get CSV string from StringIO
        csv_string = output.getvalue()
        output.close()

        return csv_string
    except Exception as e:
        print(f"Error fetching metadata with pattern from table {table_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        cursor.close()
        connection.close()

def get_table_names(connection):
    cursor = connection.cursor()
    # Filter out tables that start with underscore
    cursor.execute("SHOW TABLES")
    all_tables = cursor.fetchall()
    # Filter in Python instead of SQL to avoid escape issues
    tables = [row[0] for row in all_tables if not row[0].startswith('_')]
    cursor.close()
    return tables

def create_table(connection, table_name, data, columns):
    cursor = connection.cursor()
    try:
        # Construct the CREATE TABLE statement
        column_definitions = ", ".join([f"`{col}` TEXT" for col in columns])
        create_table_sql = f"CREATE TABLE `{table_name}` ({column_definitions})"
        cursor.execute(create_table_sql)

        # Insert data
        insert_sql = f"INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in columns])}) VALUES ({', '.join(['%s'] * len(columns))})"
        cursor.executemany(insert_sql, data)
        connection.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
    finally:
        cursor.close()

def rename_table_in_database(database, old_name, new_name):
    connection = create_connection(database)
    cursor = connection.cursor()
    try:
        # Check if the new table name already exists
        cursor.execute("SHOW TABLES LIKE %s", (new_name,))
        if cursor.fetchone():
            raise Exception(f"A table named '{new_name}' already exists.")

        # Use SQL to rename the table
        cursor.execute(f"RENAME TABLE `{old_name}` TO `{new_name}`;")
        connection.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error renaming table: {err}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

def get_total_database_size(include_system_dbs=False):
    """Calculate the total size of all databases in bytes.
    
    Args:
        include_system_dbs (bool): Whether to include system databases in the calculation
        
    Returns:
        tuple: (total_size_bytes, formatted_size)
    """
    connection = create_connection()
    if connection is None:
        return 0, "0 B"
    
    cursor = connection.cursor()
    try:
        # Define restricted databases to exclude
        restricted_dbs = ['performance_schema', 'mysql', 'information_schema', 'sys']
        
        if include_system_dbs:
            # Include all databases
            query = """
            SELECT 
                SUM(data_length + index_length) as total_size
            FROM information_schema.TABLES
            """
            cursor.execute(query)
        else:
            # Exclude system databases
            query = """
            SELECT 
                SUM(data_length + index_length) as total_size
            FROM information_schema.TABLES
            WHERE table_schema NOT IN ({})
            """.format(','.join(['%s'] * len(restricted_dbs)))
            cursor.execute(query, restricted_dbs)
        
        result = cursor.fetchone()
        total_size = result[0] if result[0] else 0
        
        # Convert bytes to human readable format
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(total_size)
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        formatted_size = f"{size:.2f} {units[unit_index]}"
        return total_size, formatted_size
        
    except mysql.connector.Error as err:
        print(f"Error calculating database size: {err}")
        return 0, "0 B"
    finally:
        cursor.close()
        connection.close()

def create_long_running_connection(database=None):
    """Create a database connection with increased timeout settings for long-running operations."""
    connection = mysql.connector.connect(
        host=DB_CONFIG['DB_HOST'],
        user=DB_CONFIG['DB_USER'],
        password=DB_CONFIG['MYSQL_PASSWORD_RAW'],
        database=database,
        connect_timeout=300,  # 5 minutes connection timeout
        connection_timeout=300,  # 5 minutes connection timeout 
        # For MySQL 8.0+, use these settings:
        # net_read_timeout=3600,  # 1 hour read timeout
        # net_write_timeout=3600,  # 1 hour write timeout
        # Older versions may need these:
        read_timeout=3600,    # 1 hour read timeout 
        write_timeout=3600    # 1 hour write timeout
    )
    
    # Set session variables for this connection
    cursor = connection.cursor()
    cursor.execute("SET SESSION wait_timeout=3600")  # 1 hour
    cursor.execute("SET SESSION max_execution_time=3600000")  # 1 hour in milliseconds
    cursor.execute("SET SESSION net_read_timeout=3600")  # 1 hour
    cursor.execute("SET SESSION net_write_timeout=3600")  # 1 hour
    cursor.execute("SET SESSION interactive_timeout=3600")  # 1 hour
    connection.commit()
    
    return connection

def create_long_running_engine(db_name):
    """Create an SQLAlchemy engine with increased timeout settings."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool
    
    connect_args = {
        'connect_timeout': 300,
        'read_timeout': 3600,
        'write_timeout': 3600
    }
    
    # Create engine with custom pool settings and connect args
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD']}@{DB_CONFIG['DB_HOST']}/{db_name}",
        pool_size=5,  # Start with 5 connections in the pool
        max_overflow=10,  # Allow up to 10 additional connections
        pool_timeout=300,  # Wait up to 5 minutes for a connection
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args=connect_args
    )
    
    return engine

def get_schema_sizes():
    """Get all schemas/databases sorted by their size in descending order.
    
    Returns:
        list: A list of dictionaries with 'schema_name', 'size_bytes', and 'formatted_size'
    """
    connection = create_connection()
    cursor = connection.cursor()
    try:
        # Define restricted databases to exclude
        restricted_dbs = ['performance_schema', 'mysql', 'information_schema', 'sys']
        
        # Query to get individual database sizes
        query = """
        SELECT 
            table_schema as 'schema_name',
            SUM(data_length + index_length) as 'total_size'
        FROM information_schema.TABLES
        WHERE table_schema NOT IN ({})
        GROUP BY table_schema
        ORDER BY total_size DESC
        """.format(','.join(['%s'] * len(restricted_dbs)))
        
        cursor.execute(query, restricted_dbs)
        results = cursor.fetchall()
        
        # Format the results
        schema_sizes = []
        for schema_name, total_size in results:
            # Convert bytes to human readable format
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            size = float(total_size)
            unit_index = 0
            while size >= 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1
            
            formatted_size = f"{size:.2f} {units[unit_index]}"
            
            schema_sizes.append({
                'schema_name': schema_name,
                'size_bytes': total_size,
                'formatted_size': formatted_size
            })
        
        return schema_sizes
        
    except mysql.connector.Error as err:
        print(f"Error calculating schema sizes: {err}")
        return []
    finally:
        cursor.close()
        connection.close()