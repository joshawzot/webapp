import mysql.connector
from db_operations import DB_CONFIG  # Consolidated database configuration

def list_databases(cursor):
    """List all databases on the MySQL server."""
    try:
        cursor.execute("SHOW DATABASES")
        print("Available databases:")
        for db in cursor:
            print(db[0])
    except mysql.connector.Error as err:
        print(f"Failed to list databases: {err}")
        exit(1)

def drop_database(cursor, db_name):
    """Drop an existing MySQL database."""
    try:
        cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
        print(f"Database {db_name} dropped successfully.")
    except mysql.connector.Error as err:
        print(f"Failed dropping database: {err}")
        exit(1)

def connect_and_perform(db_action, database_name):
    """Connect to the MySQL server and perform a specified action on a database."""
    try:
        with mysql.connector.connect(host=DB_CONFIG['DB_HOST'],
                                     user=DB_CONFIG['DB_USER'],
                                     password=DB_CONFIG['MYSQL_PASSWORD'],
                                     port=DB_CONFIG.get('RDS_PORT')) as conn, conn.cursor() as cursor:
            db_action(cursor, database_name)
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        exit(1)

def main():
    # Connect to the MySQL server and list databases
    connect_and_perform(list_databases, None)

    # Prompt user for database names to drop
    input_databases = input("Enter the names of the databases you wish to drop, separated by commas: ")
    database_names = [db.strip() for db in input_databases.split(',')]

    # Drop specified databases
    for database_name in database_names:
        connect_and_perform(drop_database, database_name)

    input("Press Enter to close the window...")

if __name__ == "__main__":
    main()
