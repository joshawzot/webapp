#!/bin/bash

# Script to run conversion with proper sudo handling
echo "🔐 Running conversion with elevated privileges..."
echo "This will convert all remaining schemas to direct SDA1 access."

# Prompt for password once at the start
sudo -v

if [ $? -eq 0 ]; then
    echo "✅ Sudo access confirmed"
    echo "🚀 Starting conversion process..."
    
    # Keep sudo alive in background
    while true; do
        sudo -n true
        sleep 60
        kill -0 "$$" || exit
    done 2>/dev/null &
    
    # Run the conversion
    sudo python3 convert_all_to_sda1.py --all
    
    echo "🎉 Conversion process completed!"
else
    echo "❌ Sudo access required for conversion"
    exit 1
fi