#!/usr/bin/env python3
"""
Check what Jupyter-related packages are installed and their versions.
This will help diagnose compatibility issues.
"""

import importlib
import subprocess
import sys

def check_package(package_name):
    """Check if a package is installed and get its version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None

def run_command(command):
    """Run a shell command and return its output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def main():
    print("=" * 60)
    print("Jupyter Environment Checker")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print("-" * 60)
    
    # Check common Jupyter-related packages
    packages = [
        "jupyter_core",
        "notebook",
        "jupyterlab",
        "jupyter_server",
        "jupyter_client",
        "jupyterlab_server",
        "jupyter_server_proxy",
        "nbconvert",
        "nbformat",
        "traitlets",
        "ipykernel",
        "ipython",
        "tornado",
        "jinja2",
    ]
    
    print("Installed packages:")
    for package in packages:
        installed, version = check_package(package)
        status = f"✓ {version}" if installed else "✗ Not installed"
        print(f"{package:25} {status}")
    
    print("-" * 60)
    
    # Check jupyter command
    print("Jupyter command:")
    jupyter_path = run_command("which jupyter")
    print(f"Path: {jupyter_path}")
    jupyter_version = run_command("jupyter --version")
    print(f"Version info:\n{jupyter_version}")
    
    print("-" * 60)
    
    # Check notebook command specifically
    print("Notebook command:")
    notebook_version = run_command("jupyter notebook --version")
    print(f"Notebook version: {notebook_version}")
    
    print("-" * 60)
    
    # Print some diagnostic tips
    print("Diagnostic tips:")
    print("1. Make sure notebook and jupyter_client versions are compatible")
    print("2. For notebook 6.x, you need jupyter_client 6.x")
    print("3. For notebook 7.x, you need jupyter_server instead of notebook")
    print("4. Try running 'jupyter notebook' directly to see detailed errors")
    print("=" * 60)

if __name__ == "__main__":
    main() 