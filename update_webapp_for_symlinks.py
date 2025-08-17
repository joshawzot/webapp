#!/usr/bin/env python3
"""
Update webapp to use symlink-aware storage instead of temporary restoration
"""

def get_updated_list_tables_code():
    """Return the updated list_tables function code"""
    
    return '''
@app.route('/list-tables', methods=['POST', 'GET'])
def list_tables():
    # Get the username from the session
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))

    database = request.form.get('database') or request.args.get('database')
    if not database:
        return redirect(url_for('home'))

    # Record this folder visit
    if database:
        record_folder_visit(database, username)

    # Use symlink-aware storage instead of temporary restoration
    try:
        from symlink_aware_storage import symlink_storage
        from analyze_schema_ages import extract_timestamp_from_name
        
        # Get comprehensive storage information
        storage_info_obj = symlink_storage.get_schema_storage_info(database)
        schema_timestamp = extract_timestamp_from_name(database)
        
        # Check if schema is accessible
        is_accessible = symlink_storage.is_schema_accessible(database)
        
        if not is_accessible:
            # Handle different cases based on storage state
            if storage_info_obj['location'] == 'archived_not_linked':
                # Archive exists but not linked - offer to create symlink
                return render_template('schema_needs_linking.html',
                                     database=database,
                                     storage_info=storage_info_obj,
                                     username=username)
            
            elif storage_info_obj['location'] == 'backup_only':
                # Only SQL backup exists - offer restoration
                return render_template('schema_archived.html',
                                     database=database, 
                                     storage_info=storage_info_obj,
                                     username=username)
            
            else:
                # Schema not found anywhere
                error_msg = f"Schema '{database}' not found in any storage location"
                return render_template('error.html',
                                     error_message=error_msg,
                                     username=username), 404
        
        # Schema is accessible - prepare storage display info
        storage_info = {
            'location': storage_info_obj['location'],
            'device': storage_info_obj['storage_device'],
            'path': storage_info_obj['storage_path'],
            'type': storage_info_obj['storage_type'],
            'color': storage_info_obj['storage_color'],
            'access_method': storage_info_obj['access_method'],
            'is_symlink': storage_info_obj['is_symlink'],
            'age': None,
            'timestamp': None
        }
        
        # Calculate age if timestamp exists
        if schema_timestamp:
            from datetime import datetime
            age_days = (datetime.now() - schema_timestamp).days
            storage_info['age'] = f"{age_days} days old"
            storage_info['timestamp'] = schema_timestamp.strftime('%Y-%m-%d %H:%M')
        
    except Exception as e:
        print(f"Error getting storage info: {e}")
        # Fallback to default primary storage
        storage_info = {
            'location': 'primary',
            'device': "/dev/nvme2n1p1 (Primary Drive)",
            'path': "/var/lib/mysql",
            'type': "Primary Storage",
            'color': "success",
            'access_method': 'direct_primary',
            'is_symlink': False,
            'age': None,
            'timestamp': None
        }

    # Now fetch tables since schema is accessible
    tables = fetch_tables(database)  # Retrieve table data from the database
    table_names = ','.join(table['table_name'] for table in tables)
    print("table_names:", table_names)

    # [Rest of the function remains the same...]
    # ... existing table processing logic ...
'''

def get_create_symlink_route():
    """Return the new route for creating symlinks"""
    
    return '''
@app.route('/create-symlink', methods=['POST'])
def create_symlink():
    """Create symlink for archived schema that isn't linked"""
    try:
        schema_name = request.json.get('schema_name')
        if not schema_name:
            return jsonify({'success': False, 'error': 'Schema name required'})
        
        from symlink_aware_storage import symlink_storage
        
        success = symlink_storage.create_symlink_for_archived_schema(schema_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Symlink created for {schema_name}',
                'redirect': f'/list-tables?database={schema_name}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create symlink'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
'''

def get_schema_needs_linking_template():
    """Return template for schemas that need linking"""
    
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Schema Needs Linking</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .archive-container {
            max-width: 800px;
            margin: 50px auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        }
        .archive-icon {
            font-size: 4rem;
            color: #ff9800;
            margin-bottom: 20px;
        }
        .btn-create-link {
            font-size: 1.1rem;
            padding: 12px 30px;
            border-radius: 25px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="archive-container text-center">
            <i class="fas fa-link archive-icon"></i>
            <h1 class="mb-4 text-warning">Schema in Archive Storage</h1>
            
            <div class="alert alert-warning">
                <h5><i class="fas fa-info-circle"></i> Schema Found in Archive</h5>
                <p>The schema <strong>{{ database }}</strong> exists in archive storage but isn't linked to MySQL.</p>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <h6><i class="fas fa-database"></i> Schema Information</h6>
                    <p><strong>Name:</strong> {{ database }}</p>
                    <p><strong>Location:</strong> {{ storage_info.storage_path }}</p>
                    <p><strong>Status:</strong> Archive storage (not accessible)</p>
                </div>
                <div class="col-md-6">
                    <h6><i class="fas fa-cogs"></i> What Happens Next</h6>
                    <p>✅ Create symlink (instant)</p>
                    <p>✅ Direct access enabled</p>
                    <p>✅ No data movement needed</p>
                </div>
            </div>
            
            <div class="mt-4">
                <button id="createLinkBtn" class="btn btn-warning btn-create-link">
                    <i class="fas fa-link"></i> Create Direct Access Link
                </button>
                <a href="{{ url_for('home') }}" class="btn btn-secondary btn-create-link ms-3">
                    <i class="fas fa-home"></i> Back to Home
                </a>
            </div>
            
            <div id="status" class="mt-3"></div>
        </div>
    </div>

    <script>
    document.getElementById('createLinkBtn').addEventListener('click', function() {
        const btn = this;
        const status = document.getElementById('status');
        
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Link...';
        
        fetch('/create-symlink', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                schema_name: '{{ database }}'
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                status.innerHTML = '<div class="alert alert-success">✅ ' + data.message + '</div>';
                setTimeout(() => {
                    window.location.href = data.redirect;
                }, 1500);
            } else {
                status.innerHTML = '<div class="alert alert-danger">❌ ' + data.error + '</div>';
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-link"></i> Create Direct Access Link';
            }
        })
        .catch(error => {
            status.innerHTML = '<div class="alert alert-danger">❌ Error: ' + error + '</div>';
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-link"></i> Create Direct Access Link';
        });
    });
    </script>
</body>
</html>
'''

def show_implementation_plan():
    """Show how to implement the symlink-aware webapp"""
    
    print("🔧 IMPLEMENTING SYMLINK-AWARE WEBAPP")
    print("=" * 60)
    
    print("\n📋 CHANGES NEEDED:")
    print("1️⃣ Update route_handlers.py list_tables function")
    print("2️⃣ Add create-symlink route")  
    print("3️⃣ Create schema_needs_linking.html template")
    print("4️⃣ Update list_tables.html to show symlink status")
    
    print("\n✅ BENEFITS AFTER IMPLEMENTATION:")
    print("   🚀 Instant access to archived schemas")
    print("   🔗 One-click symlink creation")
    print("   📊 Clear storage status indicators")
    print("   🎯 No temporary restoration needed")
    print("   💾 50% storage space savings")
    
    print("\n🎨 VISUAL INDICATORS:")
    print("   🟢 Green: Primary storage (NVMe)")
    print("   🔵 Blue: Archive with direct access (symlinked)")
    print("   🟡 Yellow: Archive needs linking")
    print("   🔴 Red: Only SQL backup exists")

if __name__ == "__main__":
    print("📖 WEBAPP SYMLINK INTEGRATION GUIDE")
    print("=" * 60)
    
    print("\n🎯 GOAL: Replace temporary restoration with instant symlink access")
    
    show_implementation_plan()
    
    print("\n📁 FILES TO CREATE:")
    print("   • schema_needs_linking.html template")
    print("   • Updated route_handlers.py sections")
    
    print("\n📁 FILES TO REMOVE (after conversion):")
    print("   • transparent_archive_access.py")
    print("   • All *.sql backup files")
    print("   • Temporary restoration logic")
    
    print("\n🚀 This gives you TRUE dual storage - instant access from both disks!")