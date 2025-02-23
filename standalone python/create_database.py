import mysql.connector
#from config import DB_CONFIG  # Use the consolidated DB configuration

# config.py
from urllib.parse import quote_plus

# Initialize DB_CONFIG
DB_CONFIG = {}

# Local mysql on admin2
DB_CONFIG['RDS_PORT'] = 3306  # Explicitly set to default MySQL port
DB_CONFIG['DB_HOST'] = "localhost"
DB_CONFIG['DB_USER'] = "root"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = 'new_password'

#for remote user on other machine
'''DB_CONFIG['RDS_PORT'] = 3306  # Implicitly defaults to 3306
DB_CONFIG['DB_HOST'] = "192.168.68.164"
DB_CONFIG['DB_USER'] = "remote_user"
DB_CONFIG['MYSQL_PASSWORD_RAW'] = '' '''

# Common for all configurations
DB_CONFIG['MYSQL_PASSWORD'] = quote_plus(DB_CONFIG['MYSQL_PASSWORD_RAW'])
def create_database(cursor, db_name):
    """Initialize a new MySQL database."""
    try:
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Database {db_name} created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        exit(1)

def create_database_setup(database_name):
    """Set up the database using configurations from DB_CONFIG."""
    try:
        # Establish connection using the settings from DB_CONFIG
        with mysql.connector.connect(host=DB_CONFIG['DB_HOST'],
                                     user=DB_CONFIG['DB_USER'],
                                     password=DB_CONFIG['MYSQL_PASSWORD'],
                                     port=DB_CONFIG.get('RDS_PORT')) as conn, conn.cursor() as cursor:
            try:
                cursor.execute(f"USE {database_name}")
            except mysql.connector.Error as err:
                print(f"Database {database_name} does not exist.")
                if err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                    create_database(cursor, database_name)
                    conn.database = database_name
                else:
                    print('Exiting')
                    print(err)
                    exit(1)
    except Exception as e:
        print('Exiting')
        print(f"Error connecting to the database: {e}")
        exit(1)

# Main code
database_names = ['param']

for database_name in database_names:
    create_database_setup(database_name)