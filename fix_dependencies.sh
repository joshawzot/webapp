#!/bin/bash

# First uninstall the problematic Werkzeug version
pip uninstall -y werkzeug

# Then install the specific version required by Flask 2.0.1
pip install -r requirements.txt

echo "Dependencies fixed. You can now run the Flask application." 