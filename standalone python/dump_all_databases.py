import subprocess
from db_operations import DB_CONFIG  # Assumes DB_CONFIG contains your database credentials

def get_databases():
    """Get a list of all databases on the MySQL server."""
    command = f"mysql -h {DB_CONFIG['DB_HOST']} -u{DB_CONFIG['DB_USER']} -p{DB_CONFIG['MYSQL_PASSWORD']} --port={DB_CONFIG.get('RDS_PORT', 3306)} -e 'SHOW DATABASES;'"
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    # Extract database names, ignoring the first line ('Database') and system databases
    databases = result.stdout.splitlines()[1:]
    return [db for db in databases if db not in ['information_schema', 'performance_schema', 'mysql', 'sys']]

def dump_database(db_name, output_dir):
    """Dump a specific database to a local .sql file."""
    output_file = f"{output_dir}/{db_name}.sql"
    command = f"mysqldump -h {DB_CONFIG['DB_HOST']} -u{DB_CONFIG['DB_USER']} -p{DB_CONFIG['MYSQL_PASSWORD']} --port={DB_CONFIG.get('RDS_PORT', 3306)} {db_name} > {output_file}"
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Database {db_name} has been dumped to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while dumping the database {db_name}: {e}")

def main():
    output_dir = input("Enter the directory to store the .sql files: ")
    databases = get_databases()
    for db in databases:
        dump_database(db, output_dir)

if __name__ == "__main__":
    main()
