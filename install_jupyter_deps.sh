#!/bin/bash

# Install required packages for Jupyter integration
pip install flask==2.0.1 flask-session==0.4.0 flask-caching==2.0.1 redis==4.3.4
pip install notebook==6.4.12 jupyterlab==3.4.8 jupyter-server-proxy==3.2.2
pip install pandas==1.5.3 numpy==1.24.3 matplotlib==3.7.1 sqlalchemy==2.0.7
pip install python-pptx==0.6.21 pillow==9.5.0 scipy==1.10.1 h5py==3.8.0

echo "Installation complete. You can now run the Flask application with the Jupyter integration." 