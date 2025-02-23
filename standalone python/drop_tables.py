import mysql.connector
from db_operations import DB_CONFIG  # Use the consolidated DB configuration

def drop_specific_tables(cursor, db_name, table_names):
    """Drop specific tables in an existing MySQL database."""
    for table_name in table_names:
        try:
            # Use backticks to handle special characters in the table name
            cursor.execute(f"USE {db_name};")
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"Table {table_name} dropped successfully.")
        except mysql.connector.Error as err:
            print(f"Failed dropping table {table_name}: {err}")

def drop_tables_setup(database_name, table_names):
    """Drop specific tables in the database using configurations from DB_CONFIG."""
    try:
        # Use DB_CONFIG for connection parameters
        with mysql.connector.connect(host=DB_CONFIG['DB_HOST'],
                                     user=DB_CONFIG['DB_USER'],
                                     password=DB_CONFIG['MYSQL_PASSWORD'],
                                     port=DB_CONFIG.get('RDS_PORT')) as conn, conn.cursor() as cursor:
            drop_specific_tables(cursor, database_name, table_names)
    except Exception as e:
        print('Exiting')
        print(f"Error connecting to the database: {e}")
        exit(1)

# Main code
database_name = input("Enter the database name: ")
input_tables = input("Enter the table names to drop, separated by commas: ")
table_names = [table.strip() for table in input_tables.split(',')]

drop_tables_setup(database_name, table_names)  # Call function to drop specific tables

input("Press Enter to close the window...")
