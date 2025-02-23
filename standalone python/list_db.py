import mysql.connector

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

def connect_and_perform(db_action):
    """Connect to the MySQL server and perform a specified action on a database."""
    try:
        with mysql.connector.connect(host="localhost",
                                     user="root",
                                     password='',  # Assuming the password is handled differently or not set
                                     port=3306) as conn, conn.cursor() as cursor:
            db_action(cursor)
    except mysql.connector.Error as e:
        print(f"Error connecting to the database: {e}")
        exit(1)

def main():
    connect_and_perform(list_databases)

if __name__ == "__main__":
    main()
