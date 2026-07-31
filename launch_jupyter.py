#!/usr/bin/env python
"""
Standalone script to launch Jupyter notebook server.
This will start a Jupyter server on port 8888 that the Flask app can connect to.
"""

import os
import sys
import subprocess
import time
import secrets
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
    return secrets.token_urlsafe(32)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.environ.get('JUPYTER_NOTEBOOK_DIR', os.path.join(base_dir, 'notebooks'))
    log_dir = os.path.join(base_dir, 'logs')
    jupyter_dir = os.path.join(base_dir, '.jupyter')
    runtime_dir = os.path.join(jupyter_dir, 'runtime')
    config_dir = os.path.join(jupyter_dir, 'config')
    data_dir = os.path.join(jupyter_dir, 'data')
    ipython_dir = os.path.join(jupyter_dir, 'ipython')

    for directory in (notebook_dir, log_dir, runtime_dir, config_dir, data_dir, ipython_dir):
        os.makedirs(directory, exist_ok=True)

    if not check_port_available(8888):
        print("ERROR: Port 8888 is already in use. Cannot start Jupyter server.")
        return 1

    token = generate_token()
    token_file = os.path.join(base_dir, '.jupyter_token')
    with open(token_file, 'w') as f:
        f.write(token)
    os.chmod(token_file, 0o600)

    log_file_path = os.path.join(log_dir, 'jupyter.log')
    log_file = open(log_file_path, 'w')
    print(f"Starting Jupyter notebook server...")
    print(f"Notebook directory: {notebook_dir}")
    print(f"Log file: {log_file_path}")

    env = os.environ.copy()
    env.update({
        'JUPYTER_RUNTIME_DIR': runtime_dir,
        'JUPYTER_CONFIG_DIR': config_dir,
        'JUPYTER_DATA_DIR': data_dir,
        'IPYTHONDIR': ipython_dir,
    })

    cmd = [
        sys.executable, '-m', 'notebook',
        f'--ServerApp.root_dir={notebook_dir}',
        '--ServerApp.ip=127.0.0.1',
        '--ServerApp.port=8888',
        '--ServerApp.open_browser=False',
        f'--ServerApp.token={token}',
    ]

    try:
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env
        )
        time.sleep(5)
        if process.poll() is not None:
            log_file.close()
            print(f"ERROR: Jupyter did not start. Check {log_file_path}")
            return 1

        print(f"Jupyter notebook server is running with PID {process.pid}")
        print("Use Ctrl+C to stop the server")
        status = {
            "pid": process.pid,
            "token": token,
            "url": "http://127.0.0.1:8888",
            "notebook_dir": notebook_dir
        }
        with open(os.path.join(base_dir, '.jupyter_status.json'), 'w') as f:
            json.dump(status, f)

        try:
            process.wait()
        except KeyboardInterrupt:
            print("Stopping Jupyter server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                print("Jupyter server did not terminate gracefully, forcing...")
                process.kill()
        finally:
            log_file.close()

        print("Jupyter server stopped")
        return 0
    except Exception as e:
        print(f"Error starting Jupyter: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 