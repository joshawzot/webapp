import mysql.connector
from urllib.parse import quote_plus

# Initialize DB_CONFIG
DB_CONFIG = {}

# Local mysql on lenovoi7
DB_CONFIG['RDS_PORT'] = 3306
DB_CONFIG['DB_HOST'] = "localhost"
DB_CONFIG['DB_USER'] = "root"
DB_CONFIG['MYSQL_PASSWORD'] = ''

#for remote user on other machine
'''DB_CONFIG['RDS_PORT'] = 3306  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "192.168.68.164"
DB_CONFIG['DB_USER'] = "remote_user"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = '' '''

# Common for all configurations
#DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])

def get_non_system_databases(cursor):
    """Get all non-system databases from MySQL server."""
    try:
        # System databases that should not be dropped
        system_dbs = {'mysql', 'information_schema', 'performance_schema', 'sys'}
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor if db[0] not in system_dbs]
        return databases
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

def connect_and_perform(db_action, database_name=None):
    """Connect to the MySQL server and perform a specified action on a database."""
    try:
        with mysql.connector.connect(host=DB_CONFIG['DB_HOST'],
                                     user=DB_CONFIG['DB_USER'],
                                     password=DB_CONFIG['MYSQL_PASSWORD'],
                                     port=DB_CONFIG.get('RDS_PORT')) as conn, conn.cursor() as cursor:
            if database_name is None:
                return db_action(cursor)
            return db_action(cursor, database_name)
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        exit(1)

def main():
    # Get list of non-system databases
    databases = connect_and_perform(get_non_system_databases)
    
    if not databases:
        print("No user databases found to remove.")
        input("Press Enter to close the window...")
        return

    print("The following databases will be removed:")
    for db in databases:
        print(f"- {db}")
    
    confirmation = input("Are you sure you want to remove these databases? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Operation cancelled.")
        input("Press Enter to close the window...")
        return

    # Drop all non-system databases
    for database_name in databases:
        connect_and_perform(drop_database, database_name)

    print("\nAll user databases have been removed successfully.")
    input("Press Enter to close the window...")

if __name__ == "__main__":
    main()
