# db_operations.py
import mysql.connector
from urllib.parse import quote_plus

# Initialize DB_CONFIG
DB_CONFIG = {}

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

connection = None

def create_connection(database=None):
    """Create a new database connection."""
    connection = mysql.connector.connect(
        host=DB_CONFIG['DB_HOST'],
        user=DB_CONFIG['DB_USER'],
        password=DB_CONFIG['MYSQL_PASSWORD_RAW'],
        database=database
    )
    return connection

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
    ORDER BY CREATE_TIME DESC;
    """
    cursor.execute(table_query, (database,))
    table_info = cursor.fetchall()

    # Dictionary to store table information with dimensions
    tables = []
    for name, time in table_info:
        # Count the number of columns
        column_query = """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s;
        """
        cursor.execute(column_query, (database, name))
        column_count = cursor.fetchone()[0]

        # Count the number of rows
        row_query = f"SELECT COUNT(*) FROM `{name}`;"
        cursor.execute(row_query)
        row_count = cursor.fetchone()[0]

        # Store table info
        tables.append({'table_name': name, 'creation_time': time, 'dimensions': f"{row_count}x{column_count}"})

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

def get_table_names(connection):
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
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

def get_total_database_size():
    """Calculate the total size of all databases in bytes, excluding system databases."""
    connection = create_connection()
    if connection is None:
        return 0, "0 B"
    
    cursor = connection.cursor()
    try:
        # Define restricted databases to exclude
        restricted_dbs = ['performance_schema', 'mysql', 'information_schema', 'sys']
        
        # Query to get database sizes
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