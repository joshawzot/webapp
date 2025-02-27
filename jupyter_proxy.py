from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, redirect, request, Response
import requests

def configure_jupyter_proxy(app):
    """
    Configure Flask app to handle Jupyter proxy requests.
    
    This sets up the app to properly handle requests that need to be forwarded
    to the Jupyter notebook server running on a separate port.
    """
    # Use ProxyFix to correctly handle the X-Forwarded headers
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Configure server name and preferred URL scheme
    app.config['SERVER_NAME'] = app.config.get('SERVER_NAME', None)
    app.config['PREFERRED_URL_SCHEME'] = 'http'
    
    original_before_request = None
    
    # If there's already a before_request handler, save it
    if hasattr(app, 'before_request_funcs') and app.before_request_funcs:
        for funcs in app.before_request_funcs.values():
            if funcs and 'before_request' in [f.__name__ for f in funcs]:
                original_before_request = next((f for f in funcs if f.__name__ == 'before_request'), None)
    
    # Register a before request handler for Jupyter endpoints
    @app.before_request
    def handle_jupyter_proxy():
        """
        Proxies requests to the Jupyter server when the path starts with /jupyter/
        """
        # Skip if this isn't a Jupyter path
        if not request.path.startswith('/jupyter/'):
            # Call the original before_request handler if it exists
            if original_before_request:
                return original_before_request()
            return None
        
        # Get the Jupyter token from the app config
        token = app.config.get('JUPYTER_TOKEN', '')
        
        # Build the target URL
        path = request.path[8:]  # Remove the /jupyter/ prefix
        jupyter_url = f'http://localhost:8888/{path}'
        
        # Add the query string and token
        query_string = request.query_string.decode('utf-8')
        if query_string:
            jupyter_url += f'?{query_string}'
            if 'token=' not in query_string:
                jupyter_url += f'&token={token}'
        else:
            jupyter_url += f'?token={token}'
        
        try:
            # Log the request for debugging
            print(f"Proxying request to Jupyter: {jupyter_url}")
            
            # Create a requests session to handle cookies and headers
            session = requests.Session()
            
            # Forward the request to the Jupyter server
            jupyter_response = session.request(
                method=request.method,
                url=jupyter_url,
                headers={key: value for key, value in request.headers if key != 'Host'},
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False,
                verify=False
            )
            
            # Create a Flask response from the Jupyter response
            response = Response(
                jupyter_response.content,
                status=jupyter_response.status_code
            )
            
            # Copy headers from Jupyter response
            for key, value in jupyter_response.headers.items():
                if key.lower() not in ('content-length', 'connection', 'content-encoding', 'transfer-encoding'):
                    response.headers[key] = value
            
            return response
            
        except requests.RequestException as e:
            print(f"Error proxying to Jupyter: {e}")
            return f"Error connecting to Jupyter notebook server: {e}", 500
    
    return app 