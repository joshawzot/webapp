from flask import Flask
from flask_session import Session
from flask_caching import Cache
import redis
from werkzeug.serving import make_server

# Initialize Flask app
app = Flask(__name__)

# Redis config
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Session config
app.config['SECRET_KEY'] = '1234'
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis_client
app.config['SESSION_PERMANENT'] = False
Session(app)

# Caching config
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_HOST'] = 'localhost'
app.config['CACHE_REDIS_PORT'] = 6379
cache = Cache(app)

# Add this line to increase the maximum allowed payload to 1GB (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 100  # 100MB

# Configure Jupyter proxy
from jupyter_proxy import configure_jupyter_proxy
app = configure_jupyter_proxy(app)

# Error handler for file size exceeding MAX_CONTENT_LENGTH
@app.errorhandler(413)
def too_large(e):
    return "File is too large! Maximum size is 100MB", 413

def run_flask(port):
    server = make_server('0.0.0.0', port, app, threaded=True)
    server.serve_forever()

# Import routes after app initialization to avoid circular imports
from route_handlers import *

if __name__ == "__main__":
    port = 3000
    run_flask(port)
