#!/usr/bin/env python
"""
Troubleshooting script for Jupyter notebook integration.
This script checks if Jupyter is properly installed and can start correctly.
"""

import sys
import os
import subprocess
import time
import socket
import requests
from pathlib import Path

def check_jupyter_installed():
    """Check if Jupyter is installed and available in PATH"""
    print("Checking if Jupyter is installed...")
    try:
        version_output = subprocess.check_output(
            ["jupyter", "--version"], 
            stderr=subprocess.STDOUT
        ).decode('utf-8')
        print(f"Jupyter is installed:\n{version_output}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error checking Jupyter installation: {e}")
        return False

def check_port_available(port=8888):
    """Check if the port is available for Jupyter to use"""
    print(f"Checking if port {port} is available...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"Port {port} is in use. Jupyter might have trouble starting.")
            return False
        else:
            print(f"Port {port} is available.")
            return True
    except Exception as e:
        print(f"Error checking port: {e}")
        return False

def test_jupyter_start():
    """Try to start a Jupyter server and verify it works"""
    print("Testing Jupyter server startup...")
    
    # Create a temporary directory for the notebook
    temp_dir = Path("/home/admin2/agate_mpw5_testing/tests/postprocess")
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Launch Jupyter server
        cmd = [
            "jupyter", "notebook",
            "--no-browser",
            "--ip=127.0.0.1",
            "--port=8889",  # Use a different port to avoid conflicts
            f"--notebook-dir={temp_dir}",
            "--NotebookApp.token=test"  # Simple token for testing
        ]
        
        print(f"Starting Jupyter with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give Jupyter time to start
        print("Waiting for Jupyter server to start...")
        time.sleep(5)
        
        # Check if the process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("Jupyter server failed to start:")
            print(f"STDOUT: {stdout.decode('utf-8')}")
            print(f"STDERR: {stderr.decode('utf-8')}")
            return False
        
        # Check if we can connect to the server
        try:
            response = requests.get("http://127.0.0.1:8889?token=test", timeout=5)
            if response.status_code == 200:
                print("Successfully connected to Jupyter server!")
                return True
            else:
                print(f"Received unexpected status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"Failed to connect to Jupyter server: {e}")
            return False
        finally:
            # Terminate the Jupyter server
            print("Terminating test Jupyter server...")
            process.terminate()
            process.wait(timeout=5)
    except Exception as e:
        print(f"Error testing Jupyter: {e}")
        return False

def check_notebook_directory():
    """Check if the notebook directory exists and has any notebook files"""
    print("Checking notebook directory...")
    notebook_dir = Path("/home/admin2/agate_mpw5_testing/tests/postprocess")
    
    if notebook_dir.exists():
        print(f"Found notebook directory at: {notebook_dir}")
        
        # Check for any .ipynb files
        notebook_files = list(notebook_dir.glob("*.ipynb"))
        if notebook_files:
            print(f"Found {len(notebook_files)} notebook file(s):")
            for notebook in notebook_files:
                print(f"  - {notebook.name}")
            return True
        else:
            print("No notebook files (.ipynb) found in the directory")
            print("Jupyter will still work - you can create notebooks from the web interface")
            return True
    else:
        print(f"Notebook directory not found: {notebook_dir}")
        return False

def main():
    """Run all checks and report results"""
    print("=" * 50)
    print("Jupyter Notebook Integration Troubleshooter")
    print("=" * 50)
    
    checks = [
        ("Jupyter Installation", check_jupyter_installed),
        ("Port Availability", check_port_available),
        ("Jupyter Startup", test_jupyter_start),
        ("Notebook Directory", check_notebook_directory)
    ]
    
    results = []
    for name, check_func in checks:
        print("\n" + "=" * 50)
        print(f"Running check: {name}")
        print("-" * 50)
        success = check_func()
        results.append((name, success))
        print("-" * 50)
        print(f"Check result: {'PASS' if success else 'FAIL'}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    all_pass = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        if not success:
            all_pass = False
        print(f"  {name}: {status}")
    
    print("\nOverall status:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print("=" * 50)
    
    if not all_pass:
        print("\nTroubleshooting tips:")
        print("1. Make sure all required packages are installed:")
        print("   pip install -r requirements.txt")
        print("2. Check if port 8888 is already in use by another application")
        print("3. Verify that the postprocess directory exists and is accessible")
        print("4. Check the logs directory for error messages")
        print("5. Try running Jupyter notebook manually to see if it works")
        print("   jupyter notebook --no-browser --notebook-dir=/home/admin2/agate_mpw5_testing/tests/postprocess")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main()) 