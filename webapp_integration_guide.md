# Webapp Integration Guide for Dual Storage

This guide explains how to integrate the dual storage system into your existing webapp.

## 🔄 Integration Steps

### 1. Update Route Handlers

Replace the database access functions in `route_handlers.py`:

```python
# OLD: Import from db_operations
from db_operations import get_all_databases, create_connection

# NEW: Import enhanced functions
from dual_storage_db_operations import (
    get_all_databases_with_storage_info, 
    get_storage_statistics,
    dual_storage,
    create_connection  # This remains the same
)
```

### 2. Update Home Route

Modify the home route to include storage information:

```python
@app.route('/')
def home():
    username = session.get('username')
    if username:
        try:
            conn = create_connection()
            cursor = conn.cursor()
            
            # OLD: databases = get_all_databases(cursor)
            # NEW: Get enhanced database info with storage locations
            databases = get_all_databases_with_storage_info(cursor)
            
            # Get storage statistics
            storage_stats = get_storage_statistics()
            
            # ... rest of the function remains the same ...
            
            return render_template('home_page.html', 
                                  databases=databases,
                                  storage_stats=storage_stats,  # NEW
                                  username=username, 
                                  recent_visits=recent_visits,
                                  disk_info=disk_info,
                                  low_disk_space=low_disk_space)
```

### 3. Update Database Stats Route

Enhance the database statistics page:

```python
@app.route('/database-stats')
def database_stats():
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # Get enhanced database info
        databases = get_all_databases_with_storage_info(cursor)
        
        # Get comprehensive storage statistics
        storage_stats = get_storage_statistics()
        
        cursor.close()
        conn.close()
        
        return render_template('database_stats_enhanced.html', 
                              databases=databases,
                              storage_stats=storage_stats)
                              
    except Exception as e:
        print(f"Error in database_stats: {e}")
        return f"Error loading database statistics: {str(e)}"
```

### 4. Add Storage Management Routes

Add new routes for storage management:

```python
@app.route('/storage-overview')
def storage_overview():
    """Display storage overview across both devices."""
    try:
        storage_stats = get_storage_statistics()
        
        return render_template('storage_overview.html', 
                              storage_stats=storage_stats)
    except Exception as e:
        return f"Error loading storage overview: {str(e)}"

@app.route('/schema-location/<schema_name>')
def get_schema_location(schema_name):
    """Get the storage location of a specific schema."""
    try:
        location = dual_storage.get_schema_location(schema_name)
        accessibility = dual_storage.check_schema_accessibility(schema_name)
        
        return jsonify({
            'schema': schema_name,
            'location': location,
            'accessible': accessibility['accessible'],
            'table_count': accessibility.get('table_count', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 5. Update Templates

#### Enhanced Database Stats Template (`database_stats_enhanced.html`)

```html
<!-- Add storage overview section -->
<div class="storage-overview">
    <h3>Storage Overview</h3>
    <div class="storage-grid">
        <div class="storage-location">
            <h4>Primary Storage (/dev/nvme0n1p3)</h4>
            <p>Schemas: {{ storage_stats.location_stats.primary.count }}</p>
            <p>Size: {{ format_size(storage_stats.location_stats.primary.total_size) }}</p>
        </div>
        <div class="storage-location">
            <h4>Secondary Storage (/dev/sda1)</h4>
            <p>Schemas: {{ storage_stats.location_stats.secondary.count }}</p>
            <p>Size: {{ format_size(storage_stats.location_stats.secondary.total_size) }}</p>
        </div>
    </div>
</div>

<!-- Enhanced database table with storage location -->
<table class="database-table">
    <thead>
        <tr>
            <th>Database Name</th>
            <th>Storage Location</th>
            <th>Age (Days)</th>
            <th>Created</th>
        </tr>
    </thead>
    <tbody>
        {% for db in databases %}
        <tr class="storage-{{ db.location }}">
            <td>{{ db.name }}</td>
            <td>
                <span class="storage-badge storage-{{ db.location }}">
                    {{ db.location.title() }}
                </span>
            </td>
            <td>{{ db.age_days or 'N/A' }}</td>
            <td>{{ db.creation_time or 'N/A' }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
```

#### Storage Overview Template (`storage_overview.html`)

```html
<div class="storage-dashboard">
    <h2>Storage Management Dashboard</h2>
    
    <div class="storage-devices">
        {% for location, info in storage_stats.storage_info.items() %}
        <div class="device-card">
            <h3>{{ info.device }}</h3>
            <div class="device-stats">
                <p>Path: {{ info.path }}</p>
                <p>Free Space: {{ format_size(info.free_space) }}</p>
                <p>Schema Count: {{ info.schema_count }}</p>
            </div>
        </div>
        {% endfor %}
    </div>
    
    <div class="migration-status">
        <h3>Migration Status</h3>
        <div class="status-grid">
            {% for location, schemas in storage_stats.schemas_by_location.items() %}
            <div class="status-item">
                <h4>{{ location.title() }}</h4>
                <p>{{ schemas|length }} schemas</p>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
```

### 6. Add CSS for Storage Indicators

Add styling to distinguish storage locations:

```css
.storage-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: bold;
}

.storage-primary {
    background-color: #28a745;
    color: white;
}

.storage-secondary {
    background-color: #007bff;
    color: white;
}

.storage-secondary_only {
    background-color: #6c757d;
    color: white;
}

.storage-unknown {
    background-color: #dc3545;
    color: white;
}

.storage-overview {
    margin: 20px 0;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 8px;
}

.storage-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.storage-location {
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 6px;
}
```

## 🧪 Testing the Integration

1. **Test Dual Access**: Use the test script to verify both storage locations are accessible
2. **Monitor Performance**: Check that database queries perform normally
3. **Verify UI**: Ensure the enhanced templates display storage information correctly

## 🔧 Maintenance

### Regular Tasks
- Monitor disk usage on both devices
- Clean up old migration backups
- Verify schema accessibility
- Update storage statistics cache

### Troubleshooting
- If schemas become inaccessible, check symbolic links
- Monitor MySQL error logs for permission issues
- Verify both storage devices are properly mounted

## 📊 Monitoring Commands

```bash
# Check storage usage
df -h /dev/nvme0n1p3 /dev/sda1

# Test dual storage access
python3 dual_storage_db_operations.py

# View migration status
python3 execute_migration.py --skip-migration --skip-setup
```