from flask import Flask
from flask_session import Session
from flask_caching import Cache
import redis
from werkzeug.serving import make_server
import json
import datetime
import threading
import time
import os
from flask import request

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
    app.run(host='0.0.0.0', port=port, threaded=True)

# Add SocketIO integration
from flask_socketio import SocketIO

# Initialize SocketIO
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=True, engineio_logger=True)

# Store active SSH connections - using machine-specific identifiers
active_ssh_sessions = {}
# Track client subscriptions to terminal sessions
client_subscriptions = {}
# Store terminal output history
terminal_history = {}
# Path to store persistent session data
SESSION_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'terminal_sessions.json')

# Load any existing session data
def load_session_data():
    try:
        if os.path.exists(SESSION_DATA_PATH):
            with open(SESSION_DATA_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading session data: {str(e)}")
        return {}

# Save session data to disk
def save_session_data():
    try:
        session_data = {
            'machine_sessions': {},
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        # Only save metadata, not actual connections
        for session_id, session in active_ssh_sessions.items():
            machine_ip = session.get('machine_ip')
            machine_user = session.get('machine_user')
            terminal_id = session.get('terminal_id')
            created_at = session.get('created_at', datetime.datetime.now().isoformat())
            
            if machine_ip and machine_user and terminal_id:
                session_data['machine_sessions'][session_id] = {
                    'machine_ip': machine_ip,
                    'machine_user': machine_user,
                    'terminal_id': terminal_id,
                    'created_at': created_at
                }
        
        with open(SESSION_DATA_PATH, 'w') as f:
            json.dump(session_data, f)
    except Exception as e:
        print(f"Error saving session data: {str(e)}")

# Terminal WebSocket handlers
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'Client connected: {client_id}')
    client_subscriptions[client_id] = set()

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    print(f'Client disconnected: {client_id}')
    
    # Get sessions that this client was subscribed to
    affected_sessions = set()
    if client_id in client_subscriptions:
        affected_sessions = client_subscriptions[client_id].copy()
        client_subscriptions.pop(client_id, None)
    
    # Update user counts for affected sessions
    for session_id in affected_sessions:
        emit_user_count_update(session_id)

@socketio.on('list_terminals')
def handle_list_terminals(data):
    client_id = request.sid
    machine_ip = data.get('machine_ip')
    machine_user = data.get('machine_user')
    
    print(f"List terminals request from {client_id} for {machine_user}@{machine_ip}")
    
    machine_sessions = []
    
    # First, quickly calculate all user counts for efficiency
    session_user_counts = {}
    for cid, sessions in client_subscriptions.items():
        for session_id in sessions:
            if session_id in active_ssh_sessions:  # Only count active sessions
                if session_id not in session_user_counts:
                    session_user_counts[session_id] = []
                
                # Add user info - get username from session if available
                username = "Unknown"
                try:
                    # Try to get from Flask session first
                    from flask import session
                    if 'username' in session:
                        username = session['username']
                    else:
                        # Fall back to other methods if session username not available
                        if hasattr(request, 'remote_addr') and request.remote_addr:
                            username = request.remote_addr
                except:
                    pass
                
                # Add to the list of users for this session
                session_user_counts[session_id].append({
                    'username': username,
                    'client_id': cid
                })
    
    # Find all sessions for this machine - use more efficient filtering
    machine_session_prefix = f"machine_{machine_ip}_{machine_user}_term"
    
    for session_id, session in active_ssh_sessions.items():
        if not session_id.startswith(machine_session_prefix):
            continue
            
        terminal_id = session.get('terminal_id', '1')
        connected_users = session_user_counts.get(session_id, [])
        
        # Check session status more efficiently
        is_active = False
        channel = session.get('channel')
        if channel:
            try:
                is_active = not channel.exit_status_ready()
            except:
                is_active = False
        
        # Get creator info directly from session data - this ensures the original creator is preserved
        creator_username = session.get('creator_username', 'Unknown')
        creator_client_id = session.get('creator_client_id', '')
        
        # Mark which users are creators vs followers
        user_list = []
        
        # First add the creator even if they're not currently connected
        creator_found = False
        
        # Check if the creator is in the currently connected users
        for user in connected_users:
            user_info = {
                'username': user['username'],
                'role': 'follower'  # Default role
            }
            
            # Check if this is the creator based on client ID
            if creator_client_id and user['client_id'] == creator_client_id:
                user_info['role'] = 'creator'
                creator_found = True
            
            user_list.append(user_info)
        
        # If creator is not connected, add them to the list with a special flag
        if not creator_found and creator_username != 'Unknown':
            user_list.append({
                'username': creator_username,
                'display_name': creator_username + " (original creator)",
                'role': 'creator',
                'not_connected': True,
                'isOriginalCreator': True
            })
        
        # IMPROVED: Determine the current_user - use whoever is actually connected
        # This should match the behavior in emit_user_count_update
        current_user = 'None'
        
        # First check if we have it stored directly on the terminal session (most reliable source)
        terminal_session = active_ssh_sessions[session_id]
        if 'current_user' in terminal_session and terminal_session['current_user']:
            current_user = terminal_session['current_user']
        elif connected_users:
            # Otherwise use the first connected user
            current_user = connected_users[0]['username']
        
        # Debug output to trace which user is being sent
        print(f"Terminal {terminal_id} list: current_user={current_user}, creator={creator_username}, connected users={len(connected_users)}")
        
        # Add session info to the list
        machine_sessions.append({
            'session_id': session_id,
            'terminal_id': terminal_id,
            'created_at': session.get('created_at', 'Unknown'),
            'active': is_active,
            'users_count': len(connected_users),
            'creator': creator_username,  # Use the stored creator username
            'connected_users': user_list,
            'current_user': current_user,  # Add the current user field - whoever is actually connected
            'session_name': session.get('session_name'),  # Include custom session name if available
        })
    
    # Sort by terminal ID
    try:
        machine_sessions.sort(key=lambda s: int(s.get('terminal_id', '1').split('_')[0]))
    except:
        machine_sessions.sort(key=lambda s: s.get('terminal_id', '1'))
    
    print(f"Sending {len(machine_sessions)} sessions to client {client_id}")
    
    # Send the result immediately
    socketio.emit('terminal_list', {
        'machine_ip': machine_ip,
        'machine_user': machine_user,
        'sessions': machine_sessions
    }, room=client_id)

@socketio.on('open_terminal')
def handle_open_terminal(data):
    client_id = request.sid
    machine_ip = data.get('machine_ip')
    machine_user = data.get('machine_user')
    terminal_id = data.get('terminal_id', '1')
    session_name = data.get('session_name')  # Get custom session name
    
    if not machine_ip or not machine_user:
        socketio.emit('terminal_error', {
            'error': 'Missing machine IP or user',
            'terminal_id': terminal_id
        }, room=client_id)
        return
    
    # Define machine-specific passwords
    machine_passwords = {
        '192.168.68.124': 'slate',
        '192.168.68.234': 'Tc4$$$',
        '192.168.68.129': 'Nuc14$$$',
        '192.168.68.206': 'Nuc6$$$',
        '192.168.68.164': '40271234',
        '192.168.68.205': '2222',
        '192.168.68.235': 'Tc5$$$',
        '192.168.68.231': 'Tc1$$$',
        '192.168.68.232': 'Tc2$$$'
    }
    
    password = machine_passwords.get(machine_ip, '')
    
    # Create a machine-specific session ID instead of client-specific
    session_id = f"machine_{machine_ip}_{machine_user}_term_{terminal_id}"
    
    # Get connecting user's username
    connecting_username = "Unknown"
    try:
        # Try to get from Flask session first
        from flask import session as flask_session
        if 'username' in flask_session:
            connecting_username = flask_session['username']
        else:
            # Fall back to other methods if session username not available
            if hasattr(request, 'remote_addr') and request.remote_addr:
                connecting_username = request.remote_addr
    except:
        pass
    
    # Check if session already exists and is active
    if session_id in active_ssh_sessions:
        session = active_ssh_sessions[session_id]
        channel = session.get('channel')
        
        if channel and not channel.exit_status_ready():
            # Session is still active
            
            # Log the connection with usernames for debugging
            original_creator = session.get('creator_username', 'Unknown')
            print(f"User {connecting_username} is connecting to session {session_id} created by {original_creator}")
            
            # Initialize client_usernames if it doesn't exist
            if 'client_usernames' not in session:
                session['client_usernames'] = {}
                
            # Store the currently connecting user's username
            session['client_usernames'][client_id] = connecting_username
            
            # NEW CODE: Update the current_user field to reflect the user who is now connected
            session['current_user'] = connecting_username
            
            # NEW CODE: Implement exclusive access - kick out any existing users
            # Find all clients currently subscribed to this session
            current_users = [cid for cid, sessions in client_subscriptions.items() 
                            if session_id in sessions and cid != client_id]
            
            # If there are any existing users, disconnect them
            if current_users:
                for existing_user in current_users:
                    # Notify the existing user that they're being disconnected
                    socketio.emit('terminal_closed', {
                        'terminal_id': terminal_id,
                        'forced': True,
                        'reason': f'Another user ({connecting_username}) has taken control of this terminal session'
                    }, room=existing_user)
                    
                    # Remove session from their subscriptions
                    if existing_user in client_subscriptions:
                        if session_id in client_subscriptions[existing_user]:
                            client_subscriptions[existing_user].remove(session_id)
                    
                    print(f"Kicked out user {existing_user} from session {session_id} due to exclusive access rule")
            
            # Now subscribe the new client to the session
            if client_id not in client_subscriptions:
                client_subscriptions[client_id] = set()
            client_subscriptions[client_id].add(session_id)
            
            # Send any history we've captured
            if session_id in terminal_history:
                socketio.emit('terminal_history', {
                    'output': terminal_history[session_id],
                    'terminal_id': terminal_id
                }, room=client_id)
            
            # Notify client that session is ready
            socketio.emit('terminal_ready', {
                'success': True,
                'terminal_id': terminal_id,
                'reused': True,
                'exclusive': True,
                'creator': original_creator,  # Send creator info to the client
                'session_name': session.get('session_name')  # Send session name if it exists
            }, room=client_id)
            
            # Emit updated user count to all connected clients for this session
            emit_user_count_update(session_id)
            
            # Broadcast new terminal creation to all clients to update session lists in real-time
            socketio.emit('terminal_created', {
                'terminal_id': terminal_id,
                'session_id': session_id,
                'machine_ip': machine_ip,
                'machine_user': machine_user,
                'session_name': session_name  # Include session name
            })
            
            print(f"Reused existing session {session_id} for client {client_id} with exclusive access")
            return
    
    # If we get here, we need to create a new session or recreate a dead one
    try:
        # Set up SSH client
        import paramiko
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(machine_ip, username=machine_user, password=password)
        
        # Get transport
        transport = ssh_client.get_transport()
        transport.set_keepalive(60)
        
        # Open channel and invoke shell
        channel = transport.open_session()
        channel.get_pty(term='xterm-256color', width=80, height=24)
        channel.invoke_shell()
        
        # Initialize history for this session
        terminal_history[session_id] = ""
        
        # Store session info with machine details
        created_at = datetime.datetime.now().isoformat()
        
        # Try to get username of creator
        creator_username = "Unknown"
        try:
            # Try to get from Flask session first
            from flask import session
            if 'username' in session:
                creator_username = session['username']
            else:
                # Fall back to other methods if session username not available
                if hasattr(request, 'remote_addr') and request.remote_addr:
                    creator_username = request.remote_addr
        except:
            pass
            
        active_ssh_sessions[session_id] = {
            'ssh_client': ssh_client,
            'channel': channel,
            'transport': transport,
            'terminal_id': terminal_id,
            'machine_ip': machine_ip,
            'machine_user': machine_user,
            'created_at': created_at,
            'creator_username': creator_username,
            'creator_client_id': client_id,
            'exclusive_access': True,  # Mark session as exclusive access
            'client_usernames': {},  # Initialize client_usernames
            'current_user': connecting_username,  # Set current_user
            'session_name': session_name  # Store custom session name if provided
        }
        
        # Add session to client's subscriptions
        if client_id not in client_subscriptions:
            client_subscriptions[client_id] = set()
        client_subscriptions[client_id].add(session_id)
        
        # Emit user count update
        emit_user_count_update(session_id)
        
        # Broadcast new terminal creation to all clients to update session lists in real-time
        socketio.emit('terminal_created', {
            'terminal_id': terminal_id,
            'session_id': session_id,
            'machine_ip': machine_ip,
            'machine_user': machine_user,
            'session_name': session_name  # Include session name
        })
        
        # Start thread to read output
        def read_output():
            while True:
                if channel.exit_status_ready():
                    break
                try:
                    if channel.recv_ready():
                        data = channel.recv(1024).decode('utf-8', errors='replace')
                        
                        # Add to history
                        if session_id in terminal_history:
                            # Keep a reasonable size buffer
                            current_history = terminal_history[session_id]
                            max_history_size = 100000  # About 100KB
                            if len(current_history) + len(data) > max_history_size:
                                # Trim the history if it gets too big
                                terminal_history[session_id] = current_history[-(max_history_size//2):] + data
                            else:
                                terminal_history[session_id] += data
                        
                        # Send to all subscribed clients
                        session_clients = [cid for cid, sessions in client_subscriptions.items() 
                                          if session_id in sessions]
                        
                        for cid in session_clients:
                            socketio.emit('terminal_output', {
                                'output': data,
                                'terminal_id': terminal_id
                            }, room=cid)
                except Exception as e:
                    # Notify all subscribed clients
                    session_clients = [cid for cid, sessions in client_subscriptions.items() 
                                      if session_id in sessions]
                    
                    for cid in session_clients:
                        socketio.emit('terminal_error', {
                            'error': str(e),
                            'terminal_id': terminal_id
                        }, room=cid)
                    break
                time.sleep(0.01)
            
            # When session ends, notify all subscribed clients
            session_clients = [cid for cid, sessions in client_subscriptions.items() 
                              if session_id in sessions]
            
            for cid in session_clients:
                socketio.emit('terminal_closed', {
                    'terminal_id': terminal_id,
                    'forced': False
                }, room=cid)
            
            # We'll let sessions time out after a period of inactivity or manual closure
            try:
                terminal_history[session_id] += "\r\n[Session ended - Terminal was closed or disconnected]"
            except:
                pass
            
            # Save sessions data periodically
            save_session_data()
        
        # Start output reader thread
        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()
        
        # Save sessions data
        save_session_data()
        
        # Emit success event
        socketio.emit('terminal_ready', {
            'success': True,
            'terminal_id': terminal_id,
            'reused': False,
            'exclusive': True,
            'session_name': session_name  # Include session name in response
        }, room=client_id)
        
        print(f"Created new session {session_id} for client {client_id} with exclusive access")
        
    except Exception as e:
        socketio.emit('terminal_error', {
            'error': str(e),
            'terminal_id': terminal_id
        }, room=client_id)
        print(f"Error opening terminal: {str(e)}")

@socketio.on('send_command')
def handle_send_command(data):
    client_id = request.sid
    terminal_id = data.get('terminal_id', '1')
    command = data.get('command', '')
    machine_ip = data.get('machine_ip')
    machine_user = data.get('machine_user')
    
    # Get session using machine-specific ID
    session_id = f"machine_{machine_ip}_{machine_user}_term_{terminal_id}"
    session = active_ssh_sessions.get(session_id)
    
    if not session or not session.get('channel'):
        socketio.emit('terminal_error', {
            'error': 'Terminal session not found or disconnected',
            'terminal_id': terminal_id
        }, room=client_id)
        return
    
    channel = session.get('channel')
    try:
        channel.send(command)
    except Exception as e:
        socketio.emit('terminal_error', {
            'error': str(e),
            'terminal_id': terminal_id
        }, room=client_id)

@socketio.on('resize_terminal')
def handle_resize_terminal(data):
    client_id = request.sid
    terminal_id = data.get('terminal_id', '1')
    cols = data.get('cols', 80)
    rows = data.get('rows', 24)
    machine_ip = data.get('machine_ip')
    machine_user = data.get('machine_user')
    
    # Get session using machine-specific ID
    session_id = f"machine_{machine_ip}_{machine_user}_term_{terminal_id}"
    session = active_ssh_sessions.get(session_id)
    
    if not session or not session.get('channel'):
        return
    
    channel = session.get('channel')
    try:
        channel.resize_pty(width=cols, height=rows)
    except Exception as e:
        print(f"Error resizing terminal: {str(e)}")

@socketio.on('close_terminal')
def handle_close_terminal(data):
    client_id = request.sid
    terminal_id = data.get('terminal_id', '1')
    machine_ip = data.get('machine_ip')
    machine_user = data.get('machine_user')
    force_close = data.get('force_close', False)
    
    # Get session using machine-specific ID
    session_id = f"machine_{machine_ip}_{machine_user}_term_{terminal_id}"
    
    # Remove this client's subscription to the session
    affected_session = None
    if client_id in client_subscriptions:
        if session_id in client_subscriptions[client_id]:
            client_subscriptions[client_id].remove(session_id)
            affected_session = session_id
    
    # Update user count if needed
    if affected_session:
        emit_user_count_update(affected_session)
    
    # Only close the actual session if force_close is true
    if force_close and session_id in active_ssh_sessions:
        session = active_ssh_sessions[session_id]
        try:
            if session.get('channel'):
                session['channel'].close()
            if session.get('transport'):
                session['transport'].close()
            if session.get('ssh_client'):
                session['ssh_client'].close()
            
            # Remove from active sessions
            del active_ssh_sessions[session_id]
            # Also remove history
            if session_id in terminal_history:
                del terminal_history[session_id]
            
            # Notify all clients that were subscribed
            session_clients = [cid for cid, sessions in client_subscriptions.items() 
                              if session_id in sessions]
            
            for cid in session_clients:
                socketio.emit('terminal_closed', {
                    'terminal_id': terminal_id,
                    'forced': True
                }, room=cid)
                
                # Also remove from their subscriptions
                if cid in client_subscriptions:
                    if session_id in client_subscriptions[cid]:
                        client_subscriptions[cid].remove(session_id)
            
            # Broadcast terminal deletion to all clients for real-time session list updates
            socketio.emit('terminal_deleted', {
                'terminal_id': terminal_id,
                'session_id': session_id,
                'machine_ip': machine_ip,
                'machine_user': machine_user
            })
            
            # Save updated session data
            save_session_data()
            
        except Exception as e:
            print(f"Error closing terminal: {str(e)}")
    else:
        # Just notify this client that they've unsubscribed
        socketio.emit('terminal_closed', {
            'terminal_id': terminal_id,
            'forced': False
        }, room=client_id)

# Helper function to emit user count updates
def emit_user_count_update(session_id):
    if session_id in active_ssh_sessions:
        # Get session info first
        session_info = active_ssh_sessions[session_id]
        creator_username = session_info.get('creator_username', 'Unknown')
        creator_client_id = session_info.get('creator_client_id', '')
        
        # Count clients subscribed to this session
        clients_connected = [client for client, sessions in client_subscriptions.items() 
                           if session_id in sessions]
        connected_users = len(clients_connected)
        
        # IMPROVED: Get the current user from the session - this will be whoever is actually connected
        # If someone else took over the terminal, this should reflect that user
        current_username = None
        
        if connected_users > 0:
            # First check if we already have a current_user set directly in the session
            if 'current_user' in session_info:
                current_username = session_info['current_user']
            else:
                # Get the actual connected client's username
                client_id = clients_connected[0]
                try:
                    # Check if we have a stored username for this client in our session info
                    if 'client_usernames' in session_info and client_id in session_info['client_usernames']:
                        current_username = session_info['client_usernames'][client_id]
                    else:
                        # Try to get username from the Flask session
                        from flask import session
                        if hasattr(request, 'environ') and 'username' in request.environ.get('flask.session', {}):
                            current_username = request.environ['flask.session']['username']
                        elif hasattr(request, 'remote_addr') and request.remote_addr:
                            current_username = request.remote_addr
                        else:
                            current_username = "User"
                except:
                    # If all else fails, use a generic name
                    current_username = client_id[:8]
                
                # Store this for future reference
                session_info['current_user'] = current_username
        
        # If no one is connected, explicitly set current_username to None
        if connected_users == 0:
            current_username = None
            session_info['current_user'] = None
        
        # Get terminal ID and machine info
        terminal_id = session_info.get('terminal_id', '1')
        machine_ip = session_info.get('machine_ip')
        machine_user = session_info.get('machine_user')
        
        if not machine_ip or not machine_user:
            return
        
        # Find all clients who might be viewing this machine
        machine_session_prefix = f"machine_{machine_ip}_{machine_user}_term"
        machine_clients = set()
        
        for client, sessions in client_subscriptions.items():
            for s in sessions:
                if s.startswith(machine_session_prefix):
                    machine_clients.add(client)
        
        # Debug output to help trace user changes
        print(f"Terminal {terminal_id} update: users={connected_users}, current_user={current_username}, creator={creator_username}")
        
        # Always force a refresh if we've updated who is shown as currently using the terminal
        should_refresh = True
        prev_count = session_info.get('last_user_count', -1)
        prev_username = session_info.get('last_username', None)
        
        if prev_count != connected_users or prev_username != current_username:
            session_info['last_user_count'] = connected_users
            session_info['last_username'] = current_username
        
        # Ensure we have a place to store client-specific data
        if 'client_usernames' not in session_info:
            session_info['client_usernames'] = {}
        
        # Store the current username for this client
        if connected_users > 0 and current_username and clients_connected[0]:
            session_info['client_usernames'][clients_connected[0]] = current_username
            
        # Make sure all clients know the correct current username
        for client in machine_clients:
            # Log for debugging
            print(f"Emitting terminal_users_update to client {client} for terminal {terminal_id}: " +
                  f"users={connected_users}, current_user={current_username}, creator={creator_username}")
            
            socketio.emit('terminal_users_update', {
                'terminal_id': terminal_id,
                'users_count': connected_users,
                'current_username': current_username,
                'creator_username': creator_username,  # Always include creator username 
                'refresh_user_list': should_refresh
            }, room=client)
        
        # Broadcast to all clients for consistency
        if session_info.get('broadcast_user_count', False) == False or should_refresh:
            session_info['broadcast_user_count'] = True
            socketio.emit('terminal_users_update', {
                'terminal_id': terminal_id,
                'users_count': connected_users,
                'current_username': current_username,
                'creator_username': creator_username,
                'refresh_user_list': True
            })

# Import routes after app initialization to avoid circular imports
from route_handlers import *

# If we're running the script directly, start the server with SocketIO
if __name__ == '__main__':
    import eventlet
    eventlet.monkey_patch()
    
    # Run the app on multiple ports (3001-3020)
    import multiprocessing
    
    def start_server(port):
        print(f"Starting server on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=True)
    
    # Create and start processes for each port
    processes = []
    for port in range(3000, 3006):  # This will run from 3001 to 3020
        process = multiprocessing.Process(target=start_server, args=(port,))
        processes.append(process)
        process.start()
        print(f"Started process for port {port}")
    
    # Wait for all processes to complete (which they won't unless interrupted)
    for process in processes:
        process.join()
else:
    # WSGI entry point - this is used by Gunicorn
    # We need to make sure Gunicorn can work with SocketIO
    from eventlet import wsgi
    import eventlet
    
    # Ensure eventlet is properly initialized
    eventlet.monkey_patch()
    
    def run_with_socketio(port):
        """Run the server with SocketIO support"""
        print(f"Starting SocketIO server on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    
    # Override the run_flask function to use SocketIO
    run_flask = run_with_socketio
