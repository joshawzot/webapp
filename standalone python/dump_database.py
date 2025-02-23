import subprocess
from db_operations import DB_CONFIG  # Assuming this contains your database credentials

def dump_database(db_name, output_file):
    """Dump a specific database to a local .sql file."""
    try:
        # Construct the command to dump the database
        command = f"mysqldump -h {DB_CONFIG['DB_HOST']} -u{DB_CONFIG['DB_USER']} -p{DB_CONFIG['MYSQL_PASSWORD']} --port={DB_CONFIG.get('RDS_PORT', 3306)} {db_name} > {output_file}"

        # Execute the command
        subprocess.run(command, shell=True, check=True)
        print(f"Database {db_name} has been dumped to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while dumping the database: {e}")

# Main code
if __name__ == "__main__":
    database_name = input("Enter the database name to dump: ")
    output_file = input("Enter the desired output file path (e.g., /path/to/output.sql): ")

    dump_database(database_name, output_file)
