#!/bin/bash
#
# Background Migration Starter
# This script runs the migration in the background with proper logging
#

echo "🚀 STARTING SAFE MYSQL MIGRATION IN BACKGROUND"
echo "=============================================="
echo "Migration will run for ~51 hours"
echo "Logs will be saved to migration_output.log"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run with sudo!"
    echo "Usage: sudo bash start_migration_background.sh"
    exit 1
fi

# Navigate to the correct directory
cd /home/admin2/webapp_2

# Start migration in background with logging and auto-confirm
echo "📋 Starting migration process..."
nohup python3 safe_mysql_migration.py --auto-confirm > migration_output.log 2>&1 &
MIGRATION_PID=$!

echo "✅ Migration started!"
echo "📊 Process ID: $MIGRATION_PID"
echo "📋 Log file: migration_output.log"
echo ""
echo "🔍 To monitor progress:"
echo "   tail -f migration_output.log"
echo ""
echo "🔍 To check if still running:"
echo "   ps aux | grep safe_mysql_migration"
echo ""
echo "⏹️  To stop migration (if needed):"
echo "   sudo kill $MIGRATION_PID"
echo ""
echo "🚀 Migration is now running in background!"

# Save PID for later reference
echo $MIGRATION_PID > migration.pid
echo "💾 PID saved to migration.pid"