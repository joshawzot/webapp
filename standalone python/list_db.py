import mysql.connector
from datetime import datetime
from decimal import Decimal
import re

def extract_timestamp_from_name(db_name):
    """Extract timestamp from database name if it exists."""
    # Look for patterns like _YYYYMMDDHHMMSS
    match = re.search(r'_(\d{14})(?:$|_)', db_name)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        except ValueError:
            pass
    
    # Look for patterns like _YYYYMMDD
    match = re.search(r'_(\d{8})(?:$|_)', db_name)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d')
        except ValueError:
            pass
    
    return None

def list_databases(cursor):
    """List all databases on the MySQL server."""
    try:
        # First get all database names and sizes
        query = """
        SELECT 
            SCHEMA_NAME,
            (SELECT SUM(data_length + index_length) 
             FROM information_schema.TABLES 
             WHERE table_schema = SCHEMA_NAME) as total_size
        FROM information_schema.SCHEMATA
        WHERE SCHEMA_NAME NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
        ORDER BY SCHEMA_NAME
        """
        cursor.execute(query)
        databases = cursor.fetchall()
        
        print("\nAvailable databases:")
        print("-" * 120)
        print(f"{'Database Name':<60} {'Size':<15} {'Creation Time':<30}")
        print("-" * 120)
        
        for (db_name, size) in databases:
            # Try to get creation time from database name
            creation_time = extract_timestamp_from_name(db_name)
            time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S") if creation_time else "N/A"
            
            # Format size to human-readable format
            if size:
                size_str = format_size(float(size))
            else:
                size_str = "Empty"
                
            print(f"{db_name:<60} {size_str:<15} {time_str:<30}")
        
        print("-" * 120)
            
    except mysql.connector.Error as err:
        print(f"Failed to list databases: {err}")
        exit(1)

def format_size(size):
    """Convert size in bytes to human readable format."""
    size = float(size)  # Ensure we're working with float
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def connect_and_perform(db_action):
    """Connect to the MySQL server and perform a specified action on a database."""
    try:
        with mysql.connector.connect(
            host="localhost",
            user="root",
            password='',  # Assuming the password is handled differently or not set
            port=3306
        ) as conn, conn.cursor() as cursor:
            db_action(cursor)
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        exit(1)

def main():
    connect_and_perform(list_databases)

if __name__ == "__main__":
    main() 