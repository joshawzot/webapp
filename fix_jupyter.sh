#!/bin/bash

# Clean up any existing installations
pip uninstall -y jupyter-server-proxy jupyter_server jupyterlab notebook

# Install compatible versions of these packages
pip install notebook==6.1.5
pip install jupyter_client==6.1.12
pip install jupyterlab==3.0.16
pip install jupyter-server-proxy==1.6.0

echo "Jupyter dependencies have been fixed. Try running the launch_jupyter.py script now." 