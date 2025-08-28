#!/usr/bin/env python3
"""
View user activity logs in a readable format
"""

import os
import json
import sys
from datetime import datetime

USER_ACTIVITY_LOG_FILE = '/home/admin2/webapp_2/user_activity.log'

def view_logs(count=20, username=None, action=None, page=None):
    """View recent log entries with optional filtering"""
    
    if not os.path.exists(USER_ACTIVITY_LOG_FILE):
        print(f"📄 No log file found at {USER_ACTIVITY_LOG_FILE}")
        return
    
    try:
        with open(USER_ACTIVITY_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("📄 Log file is empty")
            return
        
        # Parse and filter entries
        entries = []
        for line in lines:
            try:
                entry = json.loads(line.strip())
                
                # Apply filters
                if username and entry.get('username') != username:
                    continue
                if action and entry.get('action') != action:
                    continue
                if page and entry.get('page') != page:
                    continue
                
                entries.append(entry)
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line.strip()}")
        
        # Show recent entries
        recent_entries = entries[-count:] if count > 0 else entries
        
        print(f"\n📋 User Activity Log")
        if username or action or page:
            filters = []
            if username: filters.append(f"user: {username}")
            if action: filters.append(f"action: {action}")
            if page: filters.append(f"page: {page}")
            print(f"🔍 Filtered by: {', '.join(filters)}")
        
        print(f"📊 Showing {len(recent_entries)} of {len(entries)} total entries")
        print("=" * 100)
        
        for entry in recent_entries:
            timestamp = entry.get('timestamp', 'Unknown')
            username = entry.get('username', 'Unknown')
            action = entry.get('action', 'Unknown')
            page = entry.get('page', 'Unknown')
            details = entry.get('details')
            ip = entry.get('ip_address', 'Unknown')
            
            details_str = ""
            if details:
                if isinstance(details, dict):
                    details_items = []
                    for k, v in details.items():
                        if isinstance(v, list) and v:
                            details_items.append(f"{k}: {', '.join(map(str, v))}")
                        elif v:
                            details_items.append(f"{k}: {v}")
                    if details_items:
                        details_str = f" | {'; '.join(details_items)}"
                else:
                    details_str = f" | {details}"
            
            print(f"{timestamp} | {username:12} | {action:20} | {page:8} | {ip:15}{details_str}")
        
        print("=" * 100)
        
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def show_stats():
    """Show statistics about the logs"""
    
    if not os.path.exists(USER_ACTIVITY_LOG_FILE):
        print(f"📄 No log file found at {USER_ACTIVITY_LOG_FILE}")
        return
    
    try:
        with open(USER_ACTIVITY_LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("📄 Log file is empty")
            return
        
        # Parse entries and collect stats
        entries = []
        users = set()
        actions = {}
        pages = {}
        ip_addresses = {}
        
        for line in lines:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
                
                username = entry.get('username')
                action = entry.get('action')
                page = entry.get('page')
                ip_address = entry.get('ip_address')
                
                if username:
                    users.add(username)
                if action:
                    actions[action] = actions.get(action, 0) + 1
                if page:
                    pages[page] = pages.get(page, 0) + 1
                if ip_address and ip_address != 'unknown':
                    ip_addresses[ip_address] = ip_addresses.get(ip_address, 0) + 1
                    
            except json.JSONDecodeError:
                continue
        
        print(f"\n📊 Activity Log Statistics")
        print("=" * 50)
        print(f"Total entries: {len(entries)}")
        print(f"Unique users: {len(users)}")
        
        if entries:
            first_entry = entries[0].get('timestamp', 'Unknown')
            last_entry = entries[-1].get('timestamp', 'Unknown')
            print(f"Date range: {first_entry} to {last_entry}")
        
        print(f"\n👥 Users: {', '.join(sorted(users))}")
        
        print(f"\n🎯 Actions (top 10):")
        for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {action:25} | {count:5} times")
        
        print(f"\n📄 Pages:")
        for page, count in sorted(pages.items(), key=lambda x: x[1], reverse=True):
            print(f"  {page:25} | {count:5} times")
        
        if ip_addresses:
            print("\n🌐 IP Addresses:")
            for ip, count in sorted(ip_addresses.items(), key=lambda x: x[1], reverse=True):
                print(f"  {ip:15} | {count:5} times")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error reading log file: {e}")

def main():
    """Main function with command line argument handling"""
    
    if len(sys.argv) == 1:
        # Default: show recent 20 entries
        view_logs(20)
    elif sys.argv[1] == 'stats':
        show_stats()
    elif sys.argv[1] == 'help':
        print("📖 User Activity Log Viewer")
        print("Usage:")
        print("  python3 view_activity_logs.py              # Show last 20 entries")
        print("  python3 view_activity_logs.py stats        # Show statistics")
        print("  python3 view_activity_logs.py 50           # Show last 50 entries")
        print("  python3 view_activity_logs.py all          # Show all entries")
        print("  python3 view_activity_logs.py tail         # Follow log in real-time")
        print("\nLog file location: " + USER_ACTIVITY_LOG_FILE)
    elif sys.argv[1] == 'tail':
        print(f"📋 Following log file: {USER_ACTIVITY_LOG_FILE}")
        print("Press Ctrl+C to stop")
        os.system(f"tail -f {USER_ACTIVITY_LOG_FILE}")
    elif sys.argv[1] == 'all':
        view_logs(-1)  # Show all entries
    elif sys.argv[1].isdigit():
        count = int(sys.argv[1])
        view_logs(count)
    else:
        print("❌ Unknown command. Use 'help' to see available options.")

if __name__ == '__main__':
    main()


