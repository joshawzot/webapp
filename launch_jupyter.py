#!/usr/bin/env python
"""
Standalone script to launch Jupyter notebook server.
This will start a Jupyter server on port 8888 that the Flask app can connect to.
"""

import os
import sys
import subprocess
import time
import random
import string
import json
import socket
from pathlib import Path

def check_port_available(port):
    """Check if the given port is available"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0
    except:
        # If we can't check, assume it's not available
        return False

def generate_token():
    """Generate a random token for Jupyter authentication"""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.path.join(base_dir, 'postprocess')
    log_dir = os.path.join(base_dir, 'logs')
    
    # Make sure directories exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Verify notebook directory exists
    if not os.path.exists(notebook_dir):
        print(f"ERROR: Notebook directory not found: {notebook_dir}")
        return 1
    
    # Check for rwb.ipynb
    notebook_file = os.path.join(notebook_dir, 'rwb.ipynb')
    if not os.path.exists(notebook_file):
        print(f"ERROR: Notebook file not found: {notebook_file}")
        return 1
    
    # Check if port 8888 is available
    if not check_port_available(8888):
        print("ERROR: Port 8888 is already in use. Cannot start Jupyter server.")
        return 1
    
    # Generate token
    token = generate_token()
    
    # Save token to a file that Flask can read
    token_file = os.path.join(base_dir, '.jupyter_token')
    with open(token_file, 'w') as f:
        f.write(token)
    
    # Create log file
    log_file_path = os.path.join(log_dir, 'jupyter.log')
    log_file = open(log_file_path, 'w')
    
    print(f"Starting Jupyter notebook server...")
    print(f"Notebook directory: {notebook_dir}")
    print(f"Log file: {log_file_path}")
    print(f"Access token: {token}")
    
    # Check Jupyter version first - try different commands based on version
    try:
        # First try using the traditional jupyter-notebook command
        cmd = [
            'python3', '-m', 'notebook',
            f'--notebook-dir={notebook_dir}',
            '--ip=0.0.0.0',
            '--port=8888',
            '--no-browser',
            f'--NotebookApp.token={token}',
            '--NotebookApp.allow_origin=*',
            '--NotebookApp.disable_check_xsrf=True',
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Start Jupyter notebook process
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file
        )
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is not None:
            print("First launch method failed, trying fallback method...")
            log_file.close()
            
            # Try alternative - use jupyterlab instead
            log_file = open(log_file_path, 'w')  # Reopen log file
            cmd = [
                'python3', '-m', 'jupyterlab',
                f'--notebook-dir={notebook_dir}',
                '--ip=0.0.0.0',
                '--port=8888',
                '--no-browser',
                f'--ServerApp.token={token}',
                '--ServerApp.allow_origin=*',
                '--ServerApp.disable_check_xsrf=True',
            ]
            
            print(f"Running alternate command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file
            )
            
            # Wait a moment for startup
            time.sleep(5)
            
            # Check if this method worked
            if process.poll() is not None:
                print("Second launch method also failed, trying fallback method with direct jupyter command...")
                log_file.close()
                
                # Last resort - try the direct jupyter command
                log_file = open(log_file_path, 'w')  # Reopen log file
                cmd = [
                    'jupyter', 'notebook',
                    f'--notebook-dir={notebook_dir}',
                    '--ip=0.0.0.0',
                    '--port=8888',
                    '--no-browser',
                    f'--NotebookApp.token={token}',
                ]
                
                print(f"Running simple command: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file
                )
                
                # Wait a moment for startup
                time.sleep(5)
                
                # Check if this method worked
                if process.poll() is not None:
                    print("ERROR: All Jupyter launch methods failed")
                    print(f"Check the log file at {log_file_path} for details")
                    return 1
        
        # If we get here, one of the methods worked
        print("=" * 80)
        print(f"Jupyter notebook server is running with PID {process.pid}")
        print(f"Access URL: http://localhost:8888/?token={token}")
        print("Use Ctrl+C to stop the server")
        print("=" * 80)
        
        # Write info to a status file for the Flask app
        status = {
            "pid": process.pid,
            "token": token,
            "url": f"http://localhost:8888",
            "notebook_path": notebook_file
        }
        
        with open(os.path.join(base_dir, '.jupyter_status.json'), 'w') as f:
            json.dump(status, f)
        
        try:
            # Keep running until interrupted
            process.wait()
        except KeyboardInterrupt:
            print("Stopping Jupyter server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                print("Jupyter server did not terminate gracefully, forcing...")
                process.kill()
        
        print("Jupyter server stopped")
        return 0
    except Exception as e:
        print(f"Error starting Jupyter: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 