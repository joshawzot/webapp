#!/bin/bash

# Configuration variables
SOURCE_HOST="localhost"
SOURCE_USER="root"
SOURCE_PASS=""
SOURCE_PORT="3306"

TARGET_HOST="192.168.68.215"
TARGET_USER="remote_user"
TARGET_PASS=""
TARGET_PORT="3306"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display error and exit
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Check if mysqldump is installed
command -v mysqldump >/dev/null 2>&1 || error_exit "mysqldump is required but not installed"
command -v mysql >/dev/null 2>&1 || error_exit "mysql client is required but not installed"

echo "Starting database schema migration..."

# Test source connection
echo "Testing source MySQL connection..."
mysql_output=$(mysql -u root -N -e "SHOW DATABASES;" 2>&1)
if [ $? -ne 0 ]; then
    echo -e "${RED}Source MySQL Connection Error: $mysql_output${NC}"
    error_exit "Failed to connect to source MySQL server"
fi

# Test target connection with verbose output
echo "Testing target MySQL connection..."
echo "Attempting to connect to $TARGET_HOST:$TARGET_PORT as $TARGET_USER..."
target_test=$(mysql --verbose -h "$TARGET_HOST" -P "$TARGET_PORT" -u "$TARGET_USER" -e "SELECT 1;" 2>&1)
if [ $? -ne 0 ]; then
    echo -e "${RED}Target MySQL Connection Error: $target_test${NC}"
    echo -e "${YELLOW}Please ensure that:${NC}"
    echo "1. MySQL is running on the target server ($TARGET_HOST)"
    echo "2. The user '$TARGET_USER' exists and can connect from this IP"
    echo "3. The MySQL port ($TARGET_PORT) is open on the target server"
    echo ""
    echo "Run these commands on the target server ($TARGET_HOST):"
    echo "1. mysql -u root"
    echo "2. DROP USER IF EXISTS '$TARGET_USER'@'%';"
    echo "3. DROP USER IF EXISTS '$TARGET_USER'@'192.168.68.164';"
    echo "4. CREATE USER '$TARGET_USER'@'%' IDENTIFIED WITH mysql_native_password BY '';"
    echo "5. GRANT ALL PRIVILEGES ON *.* TO '$TARGET_USER'@'%' WITH GRANT OPTION;"
    echo "6. FLUSH PRIVILEGES;"
    echo "7. exit;"
    echo ""
    echo -e "${YELLOW}If you need to edit MySQL configuration:${NC}"
    echo "Ask your system administrator to:"
    echo "1. Edit /etc/mysql/mysql.conf.d/mysqld.cnf"
    echo "2. Change: bind-address = 0.0.0.0"
    echo "3. Restart MySQL service"
    echo ""
    error_exit "Failed to connect to target MySQL server"
fi

echo -e "${GREEN}Successfully connected to target server${NC}"

# Get list of all databases
databases=$(mysql -u root -N -e "SHOW DATABASES;")
if [ $? -ne 0 ]; then
    echo -e "${RED}Error getting databases: $databases${NC}"
    error_exit "Failed to retrieve database list from source server"
fi

# Print found databases
echo -e "${GREEN}Found databases:${NC}"
echo "$databases" | grep -v -E '^(information_schema|performance_schema|mysql|sys)$'
echo ""

# Exclude system databases
for db in $databases; do
    if [[ "$db" != "information_schema" && "$db" != "performance_schema" && "$db" != "mysql" && "$db" != "sys" ]]; then
        echo -e "${GREEN}Migrating schema for database: $db${NC}"
        
        # Check if database exists on target
        target_check=$(mysql -h "$TARGET_HOST" -P "$TARGET_PORT" -u "$TARGET_USER" -N -e "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME='$db';" 2>&1)
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error checking database on target: $target_check${NC}"
            error_exit "Failed to check database existence on target server"
        fi
        
        if [ ! -z "$target_check" ]; then
            echo -e "${YELLOW}Database '$db' already exists on target. Dropping it for clean migration...${NC}"
            drop_result=$(mysql -h "$TARGET_HOST" -P "$TARGET_PORT" -u "$TARGET_USER" -e "DROP DATABASE \`$db\`;" 2>&1)
            if [ $? -ne 0 ]; then
                echo -e "${RED}Error dropping database: $drop_result${NC}"
                error_exit "Failed to drop existing database on target server: $db"
            fi
        fi
        
        # Export schema only (no data) using mysqldump
        echo "Exporting schema for $db..."
        mysqldump -h "$SOURCE_HOST" -P "$SOURCE_PORT" -u "$SOURCE_USER" \
            --no-data --routines --triggers --events --set-gtid-purged=OFF "$db" > "/tmp/${db}_schema.sql"
        
        if [ $? -ne 0 ]; then
            error_exit "Failed to export schema for database: $db"
        fi
        
        # Create fresh database on target server
        echo "Creating database $db on target server..."
        create_result=$(mysql -h "$TARGET_HOST" -P "$TARGET_PORT" -u "$TARGET_USER" -e "CREATE DATABASE \`$db\`;" 2>&1)
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error creating database: $create_result${NC}"
            error_exit "Failed to create database on target server: $db"
        fi
        
        # Import schema to target server
        echo "Importing schema to target server..."
        import_result=$(mysql -h "$TARGET_HOST" -P "$TARGET_PORT" -u "$TARGET_USER" "$db" < "/tmp/${db}_schema.sql" 2>&1)
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error importing schema: $import_result${NC}"
            error_exit "Failed to import schema for database: $db"
        fi
        
        # Clean up temporary file
        rm "/tmp/${db}_schema.sql"
        
        echo -e "${GREEN}Successfully migrated schema for: $db${NC}"
    fi
done

echo -e "${GREEN}Schema migration completed successfully!${NC}" 