import mysql.connector
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import datetime
import numpy as np
import pandas as pd

# Local server details
DB_HOST = "192.168.68.164"
DB_USER = "remote_user"
MYSQL_PASSWORD_RAW = ''
DB_PORT = 3306

def validate_database_name(db_name):
    if ' ' in db_name:
        print("Database name cannot contain spaces.")
        return False
    words = db_name.split('_')
    if len(words) != 7:
        print("Database name must have exactly seven parts separated by underscores.")
        return False
    for word in words:
        if '_' in word or ' ' in word:
            print("Database name parts cannot contain underscores or spaces.")
            return False
    return True

def create_database(cursor, db_name):
    try:
        cursor.execute(f"CREATE DATABASE `{db_name}`")
        print(f"Database {db_name} created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        exit(1)

def create_database_setup(database_name):
    try:
        # Establish connection using direct parameters
        with mysql.connector.connect(host=DB_HOST,
                                     user=DB_USER,
                                     password=MYSQL_PASSWORD_RAW,
                                     port=DB_PORT) as conn, conn.cursor() as cursor:
            try:
                cursor.execute(f"USE `{database_name}`")
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

def create_db_engine(database_name):
    password_encoded = quote_plus(MYSQL_PASSWORD_RAW)
    database_url = f"mysql+mysqlconnector://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{database_name}"
    print("URL", database_url)
    engine = create_engine(database_url)
    return engine

def upload_to_db(df, table_name, DATABASE_NAME, test_start_datetime):
    if validate_database_name(DATABASE_NAME):
        # Append the date to the database name, incoming date in this format: '%Y_%m_%d %H_%M_%S'
        date_str = test_start_datetime.split(' ')[0].replace('_','')
        # database_name_with_date = f"{DATABASE_NAME}_{date_str}"
        create_database_setup(DATABASE_NAME)
    else:
        print(f"Invalid database name: {DATABASE_NAME}")

    if isinstance(df, (np.ndarray, list, dict)):
        print("Input is a numpy array or a list. Converting to DataFrame...")
        if isinstance(df, dict):
            df = pd.Series(df)
        df = pd.DataFrame(df)
    else:
        print("Input is not a numpy array or a list.")
   
    print("Creating database engine...")
    engine = create_db_engine(DATABASE_NAME)
    print(f"Database engine created for database: {DATABASE_NAME}")
   
    print(f"Uploading DataFrame to table: {table_name} in database: {DATABASE_NAME}")
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
   
    print(f"Uploaded {table_name} to the database {DATABASE_NAME}.")

# Create database folder function
if __name__ == "__main__":
    print("Please call \'upload_to_db\' function")