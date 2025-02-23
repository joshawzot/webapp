import mysql.connector
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy import inspect, MetaData, Table, Column, types
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import text

# Local server details
DB_HOST = "192.168.68.164"
DB_USER = "remote_user"
MYSQL_PASSWORD_RAW = 'new_password'
DB_PORT = 3306

def validate_database_name(db_name):
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
    engine = create_engine(database_url)
    return engine

def upload_to_db(df, table_name, DATABASE_NAME):
    if isinstance(df, (np.ndarray, list)):
        print("Input is a numpy array or a list. Converting to DataFrame...")
        df = pd.DataFrame(df)
    else:
        print("Input is not a numpy array or a list.")
    
    print("Creating database engine...")
    engine = create_db_engine(DATABASE_NAME)
    print(f"Database engine created for database: {DATABASE_NAME}")
    
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        print(f"Table {table_name} does not exist. Creating table...")
        # Create table with the structure of the DataFrame
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Table {table_name} created and data uploaded.")
    else:
        print(f"Table {table_name} exists. Adjusting table structure if necessary...")
        # Get existing columns from the table
        existing_columns = inspector.get_columns(table_name)
        existing_column_names = [col['name'] for col in existing_columns]
        # Get columns from the DataFrame
        df_column_names = df.columns.tolist()
        # Identify new columns to add to the table
        new_columns = set(df_column_names) - set(existing_column_names)
        if new_columns:
            print(f"Adding new columns to table {table_name}: {new_columns}")
            metadata = MetaData()
            metadata.reflect(bind=engine)
            table = Table(table_name, metadata, autoload_with=engine)
            with engine.connect() as conn:
                for col_name in new_columns:
                    # Infer column type based on DataFrame's dtype
                    col_type = types.String(length=255)
                    dtype = df[col_name].dtype
                    if np.issubdtype(dtype, np.integer):
                        col_type = types.Integer()
                    elif np.issubdtype(dtype, np.floating):
                        col_type = types.Float()
                    elif np.issubdtype(dtype, np.datetime64):
                        col_type = types.DateTime()
                    # Add new column to the table
                    alter_stmt = f'ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_type.compile(dialect=engine.dialect)};'
                    conn.execute(text(alter_stmt))
        else:
            print("No new columns to add.")
        # Ensure DataFrame has all columns from the table
        all_columns = existing_column_names + list(new_columns)
        for col in all_columns:
            if col not in df.columns:
                df[col] = None  # Fill missing columns with None
        # Reorder DataFrame columns to match the table
        df = df[all_columns]
        print(f"Uploading DataFrame to table: {table_name} in database: {DATABASE_NAME}")
        # Append data to the table
        df.to_sql(table_name, con=engine, if_exists='append', index=False)
        print(f"Uploaded data to {table_name} in the database {DATABASE_NAME}.")

# Example usage
# database_name = 'rwb'  # or 'param'
# table_name = 'rwb_db' if database_name == 'rwb' else 'ft_param_db'
# df = pd.DataFrame(np.random.rand(10, 5))  # Example DataFrame

rwb = pd.read_csv('rwb.csv')
upload_to_db(rwb, 'rwb_db_test2', 'rwb')

#upload_to_db(df, table_name, database_name)