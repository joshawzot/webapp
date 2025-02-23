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
    """Validate that db_name is in the format 'word_word_word_word' where each word does not contain '_' or ' '."""
    if ' ' in db_name:
        print("Database name cannot contain spaces.")
        return False
    words = db_name.split('_')
    if len(words) != 4:
        print("Database name must have exactly four parts separated by underscores.")
        return False
    for word in words:
        if '_' in word or ' ' in word:
            print("Database name parts cannot contain underscores or spaces.")
            return False
    return True

def create_database(cursor, db_name):
    """Initialize a new MySQL database."""
    try:
        cursor.execute(f"CREATE DATABASE `{db_name}`")
        print(f"Database {db_name} created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        exit(1)

def create_database_setup(database_name):
    """Set up the database using direct configuration parameters."""
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
    """Create a SQLAlchemy engine for the specified database."""
    # Adjust the password to be URL-safe
    password_encoded = quote_plus(MYSQL_PASSWORD_RAW)
    # Create the database URL
    database_url = f"mysql+mysqlconnector://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{database_name}"
    # Create and return the engine
    engine = create_engine(database_url)
    return engine

def upload_to_db(df, table_name, DATABASE_NAME):
    """Upload a DataFrame to a specified table in the database."""
    # Check if the input is a numpy.ndarray or a list and convert it to a DataFrame if necessary
    if isinstance(df, (np.ndarray, list)):
        print("Input is a numpy array or a list. Converting to DataFrame...")
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
database_names = ['Adrien_Slate_chip101_finetune']  #modify the name here
for database_name in database_names:
    if validate_database_name(database_name):
        # Append the date to the database name
        #date_str = datetime.date.today().strftime('%Y%m%d')   
        date_str = datetime.date.today().strftime('%Y%m%d%H%M%S') #adding hour, minute, second
        database_name_with_date = f"{database_name}_{date_str}"
        create_database_setup(database_name_with_date)
    else:
        print(f"Invalid database name: {database_name}")

# Upload funcction
#df is the test data we want to upload to the database. Right now this function deal with 1d or 2d array specifically 
upload_to_db(df, "finetune_array_1", database_name_with_date) # Name the filename and specify the database folder to upload to