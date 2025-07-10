# route_handlers.py
# Note: File Explorer functionality has been removed from this application
# The following routes were removed:
# - /file-explorer
# - /api/list-directorygit 
# - /api/read-file
# - /api/save-file
# - /api/create-file
# - /api/create-directory

from collections import defaultdict
from run import app, cache, redis_client
from db_operations import *
from tools_for_plots import get_full_table_data, plot_individual_points_map  # Add plot_individual_points_map to the import
from flask_caching import Cache
from conductance_calculator import run_flint_conductance_calculator, convert_table_to_conductance, get_unique_original_values_and_conductance, get_unique_original_values_and_linear_conversion, update_linear_conversion_params
from sqlalchemy import text
import pandas as pd

# Standard library imports
import os, base64, json, time
from io import BytesIO

# External libraries
import pandas as pd
import mysql.connector
from flask import Flask, request, make_response, redirect, url_for, session, send_file, render_template, render_template_string, jsonify, flash, send_from_directory
from pptx import Presentation
import zipfile
import numpy as np
import csv
import io
from PIL import Image
from sqlalchemy import create_engine
import traceback
import subprocess
import sys  # Add this import
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import threading

import re
from flask import render_template_string
import pandas as pd
from scipy.io import loadmat
import h5py
import numpy as np
from io import BytesIO
import scipy.io
import shutil
import socket
import paramiko
from datetime import datetime

# Create a lock for thread-safe plotting
#plot_lock = threading.Lock()

# Custom module imports
from generate_plot import generate_plot

#from flask_caching import Cache
#cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Add this near the top of the file, with the other imports
import json
from datetime import datetime

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploaded_files'))


# Add this after the other Redis-related code
def record_folder_visit(folder_name, username):
    """Record a folder visit in Redis for tracking recent activity"""
    try:
        # Create a record with timestamp, folder name, and username
        visit = {
            'timestamp': datetime.now().isoformat(),
            'folder_name': folder_name,
            'username': username
        }
        
        # Get the current list of recent visits from Redis
        recent_visits_json = redis_client.get('recent_folder_visits')
        if recent_visits_json:
            recent_visits = json.loads(recent_visits_json)
        else:
            recent_visits = []
        
        # Check if this exact visit already exists (same user, same folder)
        # If it does, remove it so we can add it again at the top (most recent)
        recent_visits = [v for v in recent_visits if not (v['folder_name'] == folder_name and v['username'] == username)]
        
        # Add the new visit to the beginning
        recent_visits.insert(0, visit)
        
        # Keep only the 10 most recent visits
        recent_visits = recent_visits[:10]
        
        # Save back to Redis
        redis_client.set('recent_folder_visits', json.dumps(recent_visits))
        
        return True
    except Exception as e:
        print(f"Error recording folder visit: {e}")
        return False

def extract_statistical_data(table_names, database_name, form_data):
    """Extract average, standard deviation, and BER data for CSV downloads"""
    try:
        from generate_plot import get_group_data_from_matrix, get_group_data_new_from_matrix
        from tools_for_plots import get_full_table_data
        
        # Prepare lists to store data
        avg_values = []
        std_values = []
        ber_values = []
        group_names = []
        
        # Get form data parameters
        number_of_states = form_data.get('number_of_states', 4)
        selected_groups = form_data.get('selected_groups', '')
        target_ranges = form_data.get('target_ranges', [])
        
        # Process each table
        for table_name in table_names:
            try:
                # Get data matrix for the table
                data_matrix, _ = get_full_table_data(table_name, database_name)
                
                # Process based on pattern type
                if form_data.get('state_pattern_type') == '1D':
                    # For 1D patterns
                    groups, stats, selected_groups_list = get_group_data_new_from_matrix(
                        data_matrix, selected_groups, number_of_states)
                elif form_data.get('state_pattern_type') == 'predefined':
                    # For predefined patterns, we need pattern file array
                    pattern_file_array = get_pattern_file(form_data.get('state_pattern'))
                    groups, stats, selected_groups_list = get_group_data_from_matrix(
                        data_matrix, selected_groups, pattern_file_array)
                else:
                    continue
                
                # Extract average and std values
                table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
                table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation
                
                avg_values.append(table_avg_values)
                std_values.append(table_std_values)
                
                # Calculate BER values if target ranges are available
                if target_ranges and len(target_ranges) > 0:
                    ber_table_values = []
                    for i, group in enumerate(groups):
                        if i * 2 + 1 < len(target_ranges):
                            lower_bound = target_ranges[i * 2]
                            upper_bound = target_ranges[i * 2 + 1]
                            out_of_range = sum(1 for val in group if val < lower_bound or val > upper_bound)
                            ber_ppm = (out_of_range / len(group) * 1e6) if len(group) > 0 else 0
                            ber_table_values.append(ber_ppm)
                        else:
                            ber_table_values.append(0)
                    ber_values.append(ber_table_values)
                else:
                    ber_values.append([0] * len(table_avg_values))
                
                # Set group names if not already set
                if not group_names:
                    group_names = [f"State {i}" for i in range(len(table_avg_values))]
                    
            except Exception as e:
                print(f"Error processing table {table_name}: {e}")
                continue
        
        return {
            'avg_values': avg_values,
            'std_values': std_values, 
            'ber_values': ber_values,
            'table_names': table_names,
            'group_names': group_names
        }
        
    except Exception as e:
        print(f"Error in extract_statistical_data: {e}")
        return {
            'avg_values': None,
            'std_values': None,
            'ber_values': None,
            'table_names': table_names,
            'group_names': []
        }

@app.route('/')
def home():
    username = session.get('username')
    print(username)
    if username:
        try:
            conn = create_connection()
            cursor = conn.cursor()
            databases = get_all_databases(cursor)
            
            # Retrieve recent folder visits
            recent_visits_json = redis_client.get('recent_folder_visits')
            recent_visits = []
            if recent_visits_json:
                try:
                    recent_visits = json.loads(recent_visits_json)
                except:
                    # If JSON parsing fails, start with empty list
                    recent_visits = []
            
            # Check disk space
            disk_info = get_disk_space()
            
            # Get raw free space in bytes for comparison (10GB = 10 * 1024 * 1024 * 1024 bytes)
            disk_stats = shutil.disk_usage("/")  # Use root directory, which always exists
            #disk_stats = shutil.disk_usage("/app") original
            free_space_gb = disk_stats.free / (1024 * 1024 * 1024)
            low_disk_space = free_space_gb < 10  # True if less than 10GB
                
            cursor.close()
            conn.close()
            return render_template('home_page.html', 
                                  databases=databases,
                                  username=username, 
                                  recent_visits=recent_visits,
                                  disk_info=disk_info,
                                  low_disk_space=low_disk_space)
        except mysql.connector.Error as err:
            return str(err), 500
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            session['username'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error="Please enter a username")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/create-db')
def create_db_page():
    conn = create_connection()
    cursor = conn.cursor()
    databases = get_all_databases(cursor)  # Fetch all database names
    cursor.close()
    conn.close()

    return render_template('create_db_page.html')

@app.route('/save-txt-content/<database>/<table_name>', methods=['POST'])
def save_txt_content(database, table_name):
    try:
        content = request.json['content']
        connection = create_connection(database)
        cursor = connection.cursor()
        query = f"UPDATE `{table_name}` SET content = %s WHERE content IS NOT NULL LIMIT 1"
        cursor.execute(query, (content,))
        connection.commit()
        close_connection()
        return "Content saved successfully", 200
    except mysql.connector.Error as err:
        return str(err), 400

@app.route('/list-tables', methods=['POST', 'GET'])
def list_tables():
    # Get the username from the session
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        session['database'] = request.form.get('database')

    database = session.get('database')
    
    # Record this folder visit
    if database:
        record_folder_visit(database, username)

    tables = fetch_tables(database)  # Retrieve table data from the database
    table_names = ','.join(table['table_name'] for table in tables)
    print("table_names:", table_names)

    plot_function = "None"  # This could also be dynamically set based on POST or other conditions

    import hashlib
    hash_object = hashlib.sha256()
    hash_object.update(database.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    filepath = os.path.join(UPLOAD_FOLDER, hash_hex)
    if os.path.exists(filepath):
        images = os.listdir(filepath)
    else:
        images = []

    return render_template('list_tables.html', tables=tables, table_names=table_names, database=database, plot_function=plot_function, images = images, hash_hex = hash_hex)

@app.route('/view-table/<database>/<table_name>', methods=['GET'])
def view_table(database, table_name):
    """View the content of a specific table."""
    print('database:', database)
    print('table_name:', table_name)
    try:
        connection = create_connection(database)
        cursor = connection.cursor()
        if table_name.endswith('_txt'):
            query = f"SELECT content FROM `{table_name}` LIMIT 1"
            results = fetch_data(cursor, query)
            close_connection()
            content = results[0][0] if results else ''
            print('------')
            print(table_name)
            return render_template('table_txt.html', database=database, content=content, table_name=table_name)
        else:
            query = f"SELECT * FROM `{table_name}`"
            results = fetch_data(cursor, query)
            column_names = [desc[0] for desc in cursor.description]
            close_connection()
            return render_template('table.html', results=results, column_names=column_names)
    except mysql.connector.Error as err:
        return str(err)

# Define a dictionary to map plot function names to their corresponding functions
generate_plot_functions = {
    "generate_plot": generate_plot,
}

@app.route('/render-plot/<database>/<table_name>/<plot_function>')
def render_plot(database, table_name, plot_function):
    try:
        if 'username' not in session:
            return "User not logged in", 403

        # Check if we're using session-stored tables
        if table_name == 'from_session':
            # Retrieve table names from session
            table_names = session.get('plot_tables', [])
            
            # If table names found in session, use them
            if table_names:
                table_name = ','.join(table_names)
            else:
                flash('No tables found in session. Please select tables again.', 'warning')
                return redirect(url_for('list_tables'))
                
            # Also check if we have form data stored in session
            form_data_json = session.get('plot_form_data')
            if form_data_json:
                try:
                    form_data = json.loads(form_data_json)
                    # Clear the session data to avoid reusing it accidentally
                    session.pop('plot_form_data', None)
                except json.JSONDecodeError:
                    # If can't decode, use empty form data
                    form_data = {}
            else:
                form_data = {}
        else:
            # Parse form data from query string as before
            form_data_json = request.args.get('form_data', '{}')
            try:
                form_data = json.loads(form_data_json)
            except json.JSONDecodeError:
                return "Error: Invalid form data", 400

        # Validate plot function
        plot_functions = {
            'generate_plot': generate_plot,
        }

        # Validate plot function
        plot_function_impl = plot_functions.get(plot_function)
        if plot_function_impl is None:
            return f"Invalid plot function: {plot_function}", 400

        # Generate plot data with thread isolation
        try:
            # Create a new figure manager for this request
            import matplotlib.pyplot as plt
            plt.switch_backend('Agg')
            
            if plot_function == 'generate_plot':
                # Get the raw table names for processing
                input_table_names = table_name.split(',')
                
                # Call the generate_plot function 
                (plot_data,
                 sorted_table_names,
                 sorted_table_names_100ppm,  # These values are now None (removed functionality)
                 sorted_table_names_200ppm,  # These values are now None (removed functionality)
                 sorted_table_names_500ppm,  # These values are now None (removed functionality)
                 sorted_table_names_1000ppm,  # These values are now None (removed functionality)
                 best_32,
                 best_32_with_io,
                 outlier_coordinates,
                 correlation_analysis,
                 cluster_map,
                 sigma_distances,
                 num_states,
                 table_names,
                 sigma_table,
                 sigma_points,
                 filtered_avg_values,  # Real average values
                 filtered_std_values,  # Real standard deviation values
                 filtered_ber_results,  # Real BER results from CDF analysis
                 selected_groups,
                 location_dots_map) = plot_function_impl(input_table_names, database, form_data)
                
                # Create real statistical data for CSV downloads from actual analysis
                print(f"Creating real statistical data for tables: {table_names}")
                print(f"Debug: filtered_ber_results type: {type(filtered_ber_results)}")
                print(f"Debug: filtered_ber_results length: {len(filtered_ber_results) if filtered_ber_results else 0}")
                print(f"Debug: filtered_avg_values length: {len(filtered_avg_values) if filtered_avg_values else 0}")
                print(f"Debug: filtered_std_values length: {len(filtered_std_values) if filtered_std_values else 0}")
                if filtered_ber_results and len(filtered_ber_results) > 0:
                    print(f"Debug: Sample BER result: {filtered_ber_results[0]}")
                
                if table_names and len(table_names) > 0 and filtered_avg_values and filtered_std_values:
                    # Create group names from selected_groups
                    group_names = [f"State {group}" for group in selected_groups] if selected_groups else []
                    
                    # Extract BER data from filtered_ber_results
                    # filtered_ber_results is a list of tuples: (table_name, state_transition, sigma, ?, ppm_ber, uS_value, ?)
                    ber_data_by_table = {}
                    ber_transitions = set()
                    
                    if filtered_ber_results and len(filtered_ber_results) > 0:
                        for entry in filtered_ber_results:
                            table_name = entry[0]
                            state_transition = entry[1]
                            ppm_ber = entry[4]  # PPM BER value is at index 4
                            
                            if table_name not in ber_data_by_table:
                                ber_data_by_table[table_name] = {}
                            ber_data_by_table[table_name][state_transition] = ppm_ber
                            ber_transitions.add(state_transition)
                    
                    # Sort transitions to get consistent order
                    sorted_transitions = sorted(ber_transitions)
                    print(f"Debug: Found BER transitions: {sorted_transitions}")
                    print(f"Debug: BER data by table: {ber_data_by_table}")
                    
                    # Create group names for BER (use actual transition names from data)
                    ber_group_names = sorted_transitions
                    
                    # Convert BER data to list format matching table order
                    ber_values_list = []
                    for table_name in table_names:
                        table_ber_list = []
                        if table_name in ber_data_by_table:
                            for transition in sorted_transitions:
                                ber_value = ber_data_by_table[table_name].get(transition, 0)
                                table_ber_list.append(ber_value)
                            print(f"Debug: BER values for {table_name}: {table_ber_list}")
                        else:
                            # If table not found, use zeros
                            table_ber_list = [0 for _ in range(len(sorted_transitions))]
                            print(f"Debug: Table {table_name} not found, using zeros")
                        ber_values_list.append(table_ber_list)
                    
                    # Create the data structures for CSV download using real data
                    avg_values_data = {
                        'avg_values': filtered_avg_values,
                        'group_names': group_names
                    }
                    
                    std_values_data = {
                        'std_values': filtered_std_values,
                        'group_names': group_names
                    }
                    
                    ber_values_data = {
                        'ber_values': ber_values_list,
                        'group_names': ber_group_names
                    }
                    
                    print(f"Created real data successfully:")
                    print(f"Tables: {table_names}")
                    print(f"Groups: {group_names}")
                    print(f"Real avg data: {len(filtered_avg_values)} tables x {len(filtered_avg_values[0]) if filtered_avg_values else 0} states")
                    print(f"Real std data: {len(filtered_std_values)} tables x {len(filtered_std_values[0]) if filtered_std_values else 0} states")
                    print(f"Real BER data: {len(ber_values_list)} tables")
                    
                else:
                    # No tables or no data, create empty data
                    avg_values_data = {'avg_values': [], 'group_names': []}
                    std_values_data = {'std_values': [], 'group_names': []}
                    ber_values_data = {'ber_values': [], 'group_names': []}
            else:
                plot_data = plot_function_impl(table_name.split(','), database, form_data)
                sorted_table_names = sorted_table_names_100ppm = \
                    sorted_table_names_200ppm = sorted_table_names_500ppm = \
                    sorted_table_names_1000ppm = best_32 = best_32_with_io = \
                    outlier_coordinates = correlation_analysis = cluster_map = \
                    sigma_distances = num_states = table_names = sigma_table = sigma_points = None

            # Clean up all figures created during this request
            plt.close('all')

            if plot_data is None:
                return "Failed to generate plot data", 400

            # Get the dimensions from the first table's data matrix
            first_table_name = table_name.split(',')[0]
            data_matrix, data_matrix_size = get_full_table_data(first_table_name, database)
            rows, cols = data_matrix_size
            
            # Generate individual points map
            print("Generating points map...")
            print("Outlier coordinates:", outlier_coordinates)
            points_map = plot_individual_points_map(outlier_coordinates, (rows, cols))
            print("Points map generated:", points_map is not None)

            # Debug print for template variables
            print("Template variables:")
            print("- cluster_map present:", points_map is not None)
            print("- outlier_coordinates present:", bool(outlier_coordinates))
            print("- outlier_coordinates length:", len(outlier_coordinates) if outlier_coordinates else 0)
            
            # Extract BER filter values from form_data
            ber_lower_limit = form_data.get('ber_lower_limit')
            ber_upper_limit = form_data.get('ber_upper_limit')
            print(f"BER filter values for template: lower={ber_lower_limit}, upper={ber_upper_limit}")

            if plot_function == 'generate_plot':
                # Get the initial count of selected tables before any filtering
                if table_name == 'from_session':
                    # Use the table names from session
                    initial_tables = session.get('plot_tables', [])
                    initial_table_count = len(initial_tables)
                else:
                    # Use the table names from the URL
                    initial_tables = table_name.split(',')
                    initial_table_count = len(initial_tables)
                
                return render_template('plot.html', 
                                     plot_data=plot_data, 
                                     sorted_table_names=sorted_table_names, 
                                     sorted_table_names_100ppm=sorted_table_names_100ppm,  # Now None
                                     sorted_table_names_200ppm=sorted_table_names_200ppm,  # Now None
                                     sorted_table_names_500ppm=sorted_table_names_500ppm,  # Now None
                                     sorted_table_names_1000ppm=sorted_table_names_1000ppm,  # Now None
                                     best_32=best_32,
                                     best_32_with_io=best_32_with_io,
                                     outlier_coordinates=outlier_coordinates,
                                     correlation_analysis=correlation_analysis,
                                     cluster_map=points_map,
                                     sigma_distances=sigma_distances,
                                     num_states=num_states,
                                     table_names=table_names,
                                     target_values=form_data.get('target_values', []),
                                     sigma_table=sigma_table,
                                     sigma_points=sigma_points,
                                     ber_lower_limit=ber_lower_limit,
                                     ber_upper_limit=ber_upper_limit,
                                     top_ios_count=form_data.get('top_ios_count', 32),
                                     ber_display_option=form_data.get('ber_display_option', 'top_ios'),
                                     initial_table_count=initial_table_count,
                                     filtered_table_count=len(table_names) if table_names else 0,  # Pass table counts to template
                                     data_min_value=form_data.get('data_min_value'),
                                     data_max_value=form_data.get('data_max_value'),
                                     filter_negative_values=form_data.get('filter_negative_values', False),
                                     avg_values_data=avg_values_data,
                                     std_values_data=std_values_data,
                                     ber_values_data=ber_values_data,
                                     location_dots_map=location_dots_map)
            else:
                return render_template('plot.html', plot_data=plot_data)

        except Exception as e:
            print(f"Error generating plot: {e}")
            return f"Error generating plot: {str(e)}", 500

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/download_csv/<unique_id>/<data_type>')
def download_csv2(unique_id, data_type):
    # Retrieve plot data either from cache or Redis
    cache_key = f"plot_data_{unique_id}"
    plot_data = cache.get(cache_key)
    if not plot_data:
        stored_data_json = redis_client.get(unique_id)
        if not stored_data_json:
            return "Error: Data not found", 404
        stored_data = json.loads(stored_data_json)
        database = stored_data["database"]
        table_names = stored_data["table_name"].split(',')
        form_data = stored_data["form_data"]
        plot_function = stored_data["plot_function"]
        generate_plot_function = generate_plot_functions.get(plot_function)
        if not generate_plot_function:
            return "Error: Invalid plot function selection", 400
        plot_data = generate_plot_function(table_names, database, form_data)

    # Generate CSV based on plot_data and data_type
    if data_type == "avg_std":
        avg_values, std_values, table_names, selected_groups = plot_data
        header = ["State"] + [f"{table_name}" for table_name in table_names] + ["Row Avg", "Row Std Dev"]
        table_data = [header]
        column_data = [[] for _ in table_names]

        for i, group in enumerate(selected_groups):
            row = [f"State {group}"]
            row_data = []

            for j, table_avg in enumerate(avg_values):
                avg = table_avg[i]
                row.append(f"{avg:.2f}")
                row_data.append(avg)
                column_data[j].append(avg)

            row_avg = np.mean(row_data)
            row_std = np.std(row_data)
            row.extend([f"{row_avg:.2f}", f"{row_std:.2f}"])
            table_data.append(row)

        col_avgs = [np.mean(col) for col in column_data]
        col_stds = [np.std(col) for col in column_data]
        table_data.append(["Col Avg"] + [f"{avg:.2f}" for avg in col_avgs] + ["-", "-"])
        table_data.append(["Col Std Dev"] + [f"{std:.2f}" for std in col_stds] + ["-", "-"])
        return generate_csv_response(table_data, "avg_std_data.csv")

    elif data_type in ["sigma", "ppm", "us"]:
        ber_results, _ = plot_data
        headers = ["State/Transition"] + [name for name in table_names] + ["Row Avg"]
        data_collections = [headers[:], headers[:], headers[:]]

        grouped_data = {}
        for entry in ber_results:
            key = entry[1]
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append((entry[2], entry[3], entry[4]))

        for key, values in grouped_data.items():
            rows = [[key], [key], [key]]
            for val in values:
                rows[0].append(f"{val[0]:.4f}")
                rows[1].append(f"{int(val[1])}")
                rows[2].append(f"{int(val[2])}")
            for row in rows:
                avg = np.mean([float(v) for v in row[1:]])
                row.append(f"{avg:.4f}")

            data_collections[0].append(rows[0])
            data_collections[1].append(rows[1])
            data_collections[2].append(rows[2])

        index = {"sigma": 0, "ppm": 1, "us": 2}[data_type]
        filename = f"{data_type}_data.csv"
        return generate_csv_response(data_collections[index], filename)

def generate_csv_response(data, filename):
    csv_output = StringIO()
    for row in data:
        csv_output.write(','.join(str(item) for item in row) + '\n')
    csv_output.seek(0)
    response = make_response(csv_output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/download_csv')
def download_csv():
    database = request.args.get('database')
    table_name = request.args.get('table_name')

    try:
        # Generate the CSV data
        csv_data = get_csv_from_table(database, table_name)
        if csv_data is None:
            return "Error generating CSV file", 500

        # Create a response with the CSV data as a downloadable file
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = f'attachment; filename={table_name}.csv'
        response.mimetype = 'text/csv'
        return response
    except Exception as e:
        return str(e), 500

@app.route('/download_npy')
def download_npy():
    database = request.args.get('database')
    table_name = request.args.get('table_name')
    try:
        data = get_npy_from_table(database, table_name)
        if data is None:
            return "Error retrieving data or no data available", 500

        bio = io.BytesIO(data)

        return send_file(
            bio,
            as_attachment=True,
            download_name=f'{table_name}.npy',
            mimetype='application/octet-stream'
        )
    except Exception as e:
        print(f"Download error: {e}")
        return str(e), 500

@app.route('/download_metadata_csv')
def download_metadata_csv():
    database = request.args.get('database')
    table_name = request.args.get('table_name')

    try:
        # Generate the metadata CSV data
        csv_data = get_metadata_csv_from_table(database, table_name)
        if csv_data is None:
            return "Error generating metadata CSV file", 500

        # Create a response with the CSV data as a downloadable file
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = f'attachment; filename={table_name}_metadata.csv'
        response.mimetype = 'text/csv'
        return response
    except Exception as e:
        return str(e), 500

@app.route('/download_metadata_csv_with_pattern')
def download_metadata_csv_with_pattern():
    database = request.args.get('database')
    table_name = request.args.get('table_name')
    state_pattern = request.args.get('state_pattern')

    try:
        # Generate the metadata CSV data with state pattern levels
        csv_data = get_metadata_csv_with_pattern_from_table(database, table_name, state_pattern)
        if csv_data is None:
            return "Error generating metadata CSV file with pattern", 500

        # Create a response with the CSV data as a downloadable file
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = f'attachment; filename={table_name}_metadata.csv'
        response.mimetype = 'text/csv'
        return response
    except Exception as e:
        return str(e), 500

@app.route('/download_metadata_zip_with_pattern', methods=['POST'])
def download_metadata_zip_with_pattern():
    """
    Generate ZIP file of metadata CSVs on the backend to avoid frontend memory issues
    with large datasets like 82944x78.
    """
    import zipfile
    import tempfile
    import os
    import numpy as np
    
    try:
        data = request.get_json()
        database = data.get('database')
        table_names = data.get('table_names', [])
        state_pattern = data.get('state_pattern')
        filename = data.get('filename', 'tables_metadata')
        
        print(f"Starting ZIP generation for {len(table_names)} tables with pattern {state_pattern}")
        
        # Load the state pattern file to get its dimensions
        state_pattern_file_path = os.path.join('State_pattern_files', f'{state_pattern}.npy')
        
        if not os.path.exists(state_pattern_file_path):
            error_msg = f"❌ State pattern file not found: {state_pattern}.npy"
            print(error_msg)
            return error_msg, 400
            
        try:
            pattern_array = np.load(state_pattern_file_path)
            pattern_shape = pattern_array.shape
            print(f"Pattern {state_pattern} dimensions: {pattern_shape}")
        except Exception as e:
            error_msg = f"❌ Error loading state pattern file: {str(e)}"
            print(error_msg)
            return error_msg, 400
        
        # Check dimensions of all selected tables against the pattern
        dimension_mismatches = []
        
        for table_name in table_names:
            try:
                # Get table dimensions
                table_dimensions = get_table_dimensions(database, table_name)
                print(f"Table {table_name} dimensions: {table_dimensions}")
                
                if table_dimensions != pattern_shape:
                    dimension_mismatches.append({
                        'table': table_name,
                        'table_dimensions': table_dimensions,
                        'pattern_dimensions': pattern_shape
                    })
            except Exception as e:
                error_msg = f"❌ Error checking dimensions for table {table_name}: {str(e)}"
                print(error_msg)
                return error_msg, 400
        
        # If there are dimension mismatches, return a detailed error
        if dimension_mismatches:
            error_details = []
            for mismatch in dimension_mismatches:
                error_details.append(
                    f"• Table '{mismatch['table']}': {mismatch['table_dimensions']} "
                    f"≠ Pattern '{state_pattern}': {mismatch['pattern_dimensions']}"
                )
            
            error_message = (
                f"❌ DIMENSION MISMATCH DETECTED!\n\n"
                f"The selected state pattern '{state_pattern}' has dimensions {pattern_shape}, "
                f"but the following tables have different dimensions:\n\n" +
                "\n".join(error_details) + 
                f"\n\n💡 Please:\n"
                f"• Select a state pattern that matches your table dimensions, OR\n"
                f"• Select tables that match the pattern dimensions\n\n"
                f"Available state patterns with their dimensions can be found in the State_pattern_files folder."
            )
            
            print("Dimension mismatch detected:")
            for detail in error_details:
                print(f"  {detail}")
            
            return error_message, 400
        
        # Create a temporary file for the ZIP
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            temp_zip_path = temp_zip.name
            
        # Create ZIP file on the backend
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, table_name in enumerate(table_names):
                print(f"Processing table {i+1}/{len(table_names)}: {table_name}")
                
                # Generate CSV data for this table
                csv_data = get_metadata_csv_with_pattern_from_table(database, table_name, state_pattern)
                if csv_data is not None:
                    # Add to ZIP file
                    zip_file.writestr(f"{table_name}_metadata.csv", csv_data)
                    print(f"Added {table_name}_metadata.csv to ZIP")
                else:
                    print(f"Failed to generate CSV for {table_name}")
        
        print("ZIP file generation complete")
        
        # Send the ZIP file as response
        def remove_file(response):
            try:
                os.unlink(temp_zip_path)
                print(f"Cleaned up temporary file: {temp_zip_path}")
            except Exception as e:
                print(f"Error removing temporary file: {e}")
            return response
        
        response = send_file(
            temp_zip_path,
            as_attachment=True,
            download_name=f'{filename}.zip',
            mimetype='application/zip'
        )
        
        # Clean up temp file after sending (using Flask's after_request won't work here)
        # We'll let the OS clean it up eventually, or use a cleanup job
        
        return response
        
    except Exception as e:
        print(f"Error in download_metadata_zip_with_pattern: {str(e)}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/view-plot/<database>/<table_name>/<plot_function>', methods=['GET', 'POST'])
def view_plot(database, table_name, plot_function):
    print("view_plot")
    
    # Check if we're using session-stored tables
    if table_name == 'from_session':
        # Retrieve table names from session
        table_names = session.get('plot_tables', [])
        
        # If table names found in session, use them
        if table_names:
            table_name = ','.join(table_names)
        else:
            flash('No tables found in session. Please select tables again.', 'warning')
            return redirect(url_for('list_tables'))
    
    if request.method == "POST":
        print("POST:::::::::::::::::::::::::::::::::")
        
        # Check if this is a plot function choice submission
        plot_function_choice = request.form.get('plot_choice')
        if plot_function_choice:
            plot_function = plot_function_choice
            if plot_function in generate_plot_functions:
                if plot_function == "generate_plot":
                    return render_template('input_form_generate_plot.html', database=database, table_name=table_name, plot_function=plot_function)
            else:
                return f"Invalid plot function selection", 400
        
        # For other POST requests (form submissions), redirect to the process-plot-form endpoint
        # This will avoid URI length issues when there are many tables
        
        # We need to copy all form data to the new request
        form_data = request.form.to_dict(flat=False)
        
        # Create a form for POST submission
        form_html = '<form id="redirectForm" action="/process-plot-form" method="POST">'
        form_html += f'<input type="hidden" name="database" value="{database}">'
        form_html += f'<input type="hidden" name="table_name" value="{table_name}">'
        form_html += f'<input type="hidden" name="plot_function" value="{plot_function}">'
        
        # Add all form fields
        for key, values in form_data.items():
            for value in values:
                form_html += f'<input type="hidden" name="{key}" value="{value}">'
                
        form_html += '</form>'
        form_html += '<script>document.getElementById("redirectForm").submit();</script>'
        
        return form_html
    else:
        table_names = table_name.split(',')
        print(table_names)
        print("GET:::::::::::::::::::::::::::::::::")
        
        # Check if we're using conductance or linear conversion
        using_conductance = session.get('using_conductance', False)
        conductance_comparison = session.get('conductance_comparison', None)
        using_linear_conversion = session.get('using_linear_conversion', False)
        linear_conversion_comparison = session.get('linear_conversion_comparison', None)
        
        # Get current linear conversion parameters for the banner
        from conductance_calculator import LINEAR_CONVERSION
        linear_min = LINEAR_CONVERSION["output_min"]
        linear_max = LINEAR_CONVERSION["output_max"]
        
        return render_template('choose_plot_function_form.html', 
                              database=database, 
                              table_name=table_name,
                              using_conductance=using_conductance,
                              conductance_comparison=conductance_comparison,
                              using_linear_conversion=using_linear_conversion,
                              linear_conversion_comparison=linear_conversion_comparison,
                              linear_min=linear_min,
                              linear_max=linear_max)

@app.route('/set-conductance-values/<database>/<table_name>')
def set_conductance_values(database, table_name):
    """Route to display the form for setting conductance calculation values."""
    return render_template('set_conductance_values.html', 
                          database=database, 
                          table_name=table_name)

@app.route('/calculate-conductance/<database>/<table_name>', methods=['POST'])
def calculate_conductance(database, table_name):
    """Route to calculate conductance values based on form inputs."""
    try:
        # Get the form data
        input_params = {
            'Observed_VCM_BUF_on_UGB33': float(request.form.get('Observed_VCM_BUF_on_UGB33')),
            'Observed_VREF_BUF_on_UGB3': float(request.form.get('Observed_VREF_BUF_on_UGB3')),
            'Observed_BL_FB_on_UGB11': float(request.form.get('Observed_BL_FB_on_UGB11')),
            'Observed_1p1V_LDO_output_on_UGB33': float(request.form.get('Observed_1p1V_LDO_output_on_UGB33')),
            'UGB33_offset': float(request.form.get('UGB33_offset')),
            'UGB11_offset': float(request.form.get('UGB11_offset')),
            'ADC_offset': float(request.form.get('ADC_offset')),
            'Setting_for_BLDRV_BL_R': int(request.form.get('Setting_for_BLDRV_BL_R')),
            'Setting_for_BLDRV_BLEED_RD': int(request.form.get('Setting_for_BLDRV_BLEED_RD')),
            'Observed_ADC_Output_code': 0  # Placeholder, will be replaced for each actual value
        }
        
        # Store the conductance parameters in the session
        session['conductance_params'] = input_params
        session['using_conductance'] = True
        
        # Get original data for tables to create comparison
        table_names = table_name.split(',')
        comparison_tables = []
        
        for single_table in table_names:
            data_matrix, _ = get_full_table_data(single_table, database)
            comparison = get_unique_original_values_and_conductance(data_matrix, input_params)
            comparison_tables.extend(comparison)
        
        # Remove duplicates and sort by original value
        unique_comparison = []
        seen = set()
        for item in comparison_tables:
            if item['original'] not in seen:
                unique_comparison.append(item)
                seen.add(item['original'])
        
        unique_comparison.sort(key=lambda x: x['original'])
        
        # Store the comparison in the session
        session['conductance_comparison'] = unique_comparison
        
        # Add a flash message for user feedback
        flash('Precise conversion enabled. Precise Conductance values will be used for plotting.', 'success')
        
        # Redirect back to the Choose Plot Function page
        return redirect(f'/view-plot/{database}/{table_name}/choose')
        
    except Exception as e:
        print(f"Error calculating conductance: {str(e)}")
        flash(f"Error calculating conductance: {str(e)}", 'danger')
        return redirect(f'/view-plot/{database}/{table_name}/choose')

@app.route('/upload-file', methods=['POST'])
def upload_file():   #auto upload
    print("upload_file()")
    print("request.form:", request.form)
    print("request.files:", request.files)

    if 'db_name' not in request.form:
        return "No database selected", 400

    if 'files[]' not in request.files:
        print("No files part in request.files")  # Log missing files part
        return "No files part in the request", 400

    files = request.files.getlist('files[]')
    print("All files:", files)  # Log all files before filtering

    files = [f for f in files if f.filename]
    if not files:
        print("No files detected")  # Log no files detected
        return "No files detected", 400

    for file in files:
        print(f"File: {file.filename}, MIME Type: {file.mimetype}")

    db_name = request.form['db_name']
    engine = create_db_engine(db_name)

    results = []
    for file in files:
        filename = sanitize_table_name(file.filename)
        file_extension = filename.rpartition('_')[-1]
        print("file_extension:", file_extension)
        file_stream = BytesIO(file.read())

        try:
            df = process_file(file_stream, file_extension, db_name)
            print("Original DataFrame shape (rows, columns):", df.shape) #1296,1024

            # Final check before uploading
            if df.isnull().values.any():
                print("DataFrame contains NaN values before uploading.")
            else:
                print("DataFrame does not contain NaN values before uploading.")

            if not df.empty:
                #df = df.head(500)
                #df = df.iloc[:, :500] #1296,500

                print("DataFrame shape (rows, columns):", df.shape)
                df.to_sql(filename, engine, if_exists='replace', index=False)

                results.append(f"{filename} uploaded successfully")
            else:
                results.append(f"No data to upload for {filename}. Dataframe is empty.")
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            results.append(error_msg)

    return jsonify(results=results)

@app.route('/delete-record/<database>/<table_name>', methods=['DELETE'])  # delete a table
def delete_record(database, table_name):
    try:
        connection = create_connection(database)
        cursor = connection.cursor()

        query = f"DROP TABLE `{table_name}`"
        cursor.execute(query)

        connection.commit()
        close_connection()

        return "Record deleted successfully", 200
    except mysql.connector.Error as err:
        return str(err), 400

@app.route('/delete-records/<database>', methods=['DELETE'])  # delete multiple tables
def delete_records(database):
    try:
        # Check if request.json exists
        if not request.json or 'tables' not in request.json:
            return jsonify({'message': 'Missing "tables" data in request body'}), 400
        
        tables = request.json['tables']
        
        # Validate tables is a list
        if not isinstance(tables, list) or len(tables) == 0:
            return jsonify({'message': 'Tables must be a non-empty list'}), 400
        
        # Use a long-running connection for potentially large operations
        try:
            from db_operations import create_long_running_connection
            connection = create_long_running_connection(database)

            cursor = connection.cursor()
            
            successful_drops = []
            failed_drops = []

            for table_name in tables:
                try:
                    # Check if we're dealing with a partitioned table (view + part tables)
                    is_partitioned = False
                    
                    # First check if this is a view
                    try:
                        cursor.execute(f"SHOW CREATE VIEW `{table_name}`")
                        view_data = cursor.fetchone()
                        if view_data:
                            is_partitioned = True
                            # It's a view - drop it first
                            cursor.execute(f"DROP VIEW `{table_name}`")

                            connection.commit()
                            
                            # Now check for associated part tables
                            part_pattern = f"{table_name}_part%"
                            cursor.execute(f"SHOW TABLES LIKE '{part_pattern}'")
                            part_tables = [row[0] for row in cursor.fetchall()]
                            
                            # Drop all part tables
                            for part_table in part_tables:
                                cursor.execute(f"DROP TABLE `{part_table}`")
                                connection.commit()
                            
                            successful_drops.append(table_name)
                    except Exception as view_error:
                        # Not a view, or error checking - continue to normal table handling
                        pass
                    
                    if not is_partitioned:
                        # Regular table drop
                        cursor.execute(f"DROP TABLE `{table_name}`")
                        connection.commit()
                        successful_drops.append(table_name)
                    
                except Exception as drop_error:
                    # Track failed tables but continue with others
                    failed_drops.append({"table": table_name, "error": str(drop_error)})
                    print(f"Error dropping table {table_name}: {drop_error}")
            
            # Close the connection
            cursor.close()
            connection.close()
            
            # Return appropriate response based on results
            if failed_drops:
                return jsonify({
                    'message': f'Partially successful: Dropped {len(successful_drops)} tables, {len(failed_drops)} failed.',
                    'successful': successful_drops,
                    'failed': failed_drops
                }), 207  # 207 Multi-Status
            else:
                return jsonify({
                    'message': f'Successfully deleted {len(successful_drops)} tables.',
                    'successful': successful_drops
                }), 200
                
        except Exception as conn_error:
            # Handle database connection errors
            return jsonify({'message': f'Database connection error: {str(conn_error)}'}), 500
    
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Error in delete_records: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'An error occurred: {str(e)}'}), 500

from datetime import datetime
from flask import request

@app.route('/create-database', methods=['POST'])
def create_database():
    user_name = request.form.get('userName')
    device_info = request.form.get('deviceInfo')
    chip_info = request.form.get('chipInfo')
    macro_info = request.form.get('macroInfo')
    commit_info = request.form.get('commitInfo')
    description = request.form.get('descriptionOfTest')
    date_created = datetime.now().strftime("%Y%m%d%H%M%S")  

    # Validate that all required fields are present
    if not all([user_name, device_info, chip_info, macro_info, commit_info, description]):
        return jsonify({'message': 'All fields are required.'}), 400

    db_name = f"{user_name}_{device_info}_{chip_info}_{macro_info}_{commit_info}_{description}_{date_created}"

    # Attempt to create the database
    if create_db(db_name):
        return jsonify({'message': f"Database '{db_name}' created successfully."}), 200
    else:
        return jsonify({'message': f"Failed to create Database '{db_name}'."}), 500

@app.route('/download_pptx', methods=['POST'])
def download_pptx():
    try:
        from pptx import Presentation
        from pptx.util import Inches
        from io import BytesIO
        import base64

        # Get the plots data from the request
        data = request.get_json()
        plots = data.get('plots', [])

        # Create a new presentation
        prs = Presentation()

        for i, plot_data in enumerate(plots):
            # Add a slide with a blank layout
            slide_layout = prs.slide_layouts[6]  # Blank layout
            slide = prs.slides.add_slide(slide_layout)

            # Extract base64 image data
            image_data = plot_data.split(',')[1]
            image_binary = base64.b64decode(image_data)

            # Add image to slide
            image_stream = BytesIO(image_binary)
            slide.shapes.add_picture(image_stream, Inches(0.5), Inches(0.5), width=Inches(9), height=Inches(6.5))

        # Save the presentation to a BytesIO object
        ppt_io = BytesIO()
        prs.save(ppt_io)
        ppt_io.seek(0)

        return send_file(ppt_io, as_attachment=True, download_name='plots.pptx', mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation')

    except Exception as e:
        print(f"Error creating PPTX: {e}")
        return f"Error creating PPTX: {str(e)}", 500

@app.route('/download_csv_table', methods=['POST'])
def download_csv_table():
    try:
        from io import StringIO
        
        data = request.get_json()
        table_type = data.get('table_type')
        csv_data = data.get('csv_data')
        filename = data.get('filename', 'table_data.csv')
        
        if not csv_data:
            return "Error: No CSV data provided", 400
            
        # Create CSV content
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers and data
        for row in csv_data:
            writer.writerow(row)
        
        # Get CSV string
        csv_content = output.getvalue()
        output.close()
        
        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        print(f"Error creating CSV: {e}")
        return f"Error creating CSV: {str(e)}", 500

@app.route('/getDatabases', methods=['GET'])
def get_databases():
    # Create a database connection
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # Use your function to get database names
        databases = get_all_databases(cursor)
        
        # Don't forget to close the cursor and connection when done
        cursor.close()
        conn.close()
        
        # Return the list of databases as a JSON response
        return jsonify(databases)

    except mysql.connector.Error as err:
        # In case of any database connection errors, return an error message
        return jsonify({"error": str(err)}), 500

@app.route('/add-item', methods=['POST'])
def create_item():
    item_id = request.form['item_id']
    item_data = request.form['item_data']
    try:
        response = table.put_item(
            Item={
                'ID': item_id,
                'item_data': item_data
            }
        )
        return redirect(url_for('home'))
    except ClientError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/items', methods=['GET'])
def list_items():
    try:
        response = table.scan()
        items = response['Items']
        return jsonify(items)
    except ClientError as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_zip', methods=['POST'])
def generate_zip():
    data = request.get_json()
    table_names = data.get('tableNames', [])

    if not table_names:
        return 'No tables selected', 400

    # Create a ZIP file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for table_name in table_names:
            # Generate CSV content for each table (replace with your logic)
            df = pd.read_csv(f'data/{table_name}.csv')  # Replace with your actual data source
            csv_content = df.to_csv(index=False)
            zf.writestr(f'{table_name}.csv', csv_content)

    memory_file.seek(0)
    return send_file(memory_file, attachment_filename='tables.zip', as_attachment=True)

def rename_duplicate_columns(df):
    """
    Rename duplicate columns in a dataframe to ensure all column names are unique.
    If a column is already named like 'name_1', it will get a higher suffix number.
    Handles both string and non-string (e.g., numeric) column names.
    """
    # Get list of all column names
    columns = list(df.columns)
    seen = {}
    new_columns = []
    
    for col in columns:
        # Convert column name to string if it's not already
        col_str = str(col)
        base_name = col_str
        
        # Check if column name already contains a suffix (like 'name_1')
        suffix_match = re.search(r'^(.+)_(\d+)$', col_str)
        if suffix_match:
            base_name = suffix_match.group(1)
        
        # Track how many times we've seen this base name
        if base_name in seen:
            seen[base_name] += 1
            new_col = f"{base_name}_{seen[base_name]}"
            # Ensure uniqueness by incrementing suffix until we find an unused name
            while new_col in new_columns:
                seen[base_name] += 1
                new_col = f"{base_name}_{seen[base_name]}"
            new_columns.append(new_col)
        else:
            seen[base_name] = 0
            if col_str in new_columns:  # Check if even the first column needs to be renamed
                seen[base_name] = 1
                new_columns.append(f"{base_name}_1")
            else:
                new_columns.append(col_str)
    
    # Debug log the column renaming
    print(f"DEBUG: Renamed columns from {columns} to {new_columns}")
    
    # Assign new column names to dataframe
    df.columns = new_columns
    return df

@app.route('/mergeTablesInput', methods=['POST'])
def merge_tables_input():
    database = request.form.get('database')
    table_names = request.form.getlist('tableNames')
    if not database or not table_names:
        return 'Database or table names not provided', 400
    return render_template('input_form_merge.html', database=database, table_names=table_names)

'''Original array:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

Pattern array:
[[3 1 4 2]
 [2 4 1 3]
 [4 3 2 1]]

Flattened array:
[ 1  2  3  4  5  6  7  8  9 10 11 12]

Flattened pattern:
[3 1 4 2 2 4 1 3 4 3 2 1]

Pattern indices (argsort result):
[ 1  6 11  3  4 10  0  7  9  2  5  8]

Reordered array:
[ 2  7 12  4  5 11  1  8 10  3  6  9]

Reshaped to (a*b, 1):
[[ 2]
 [ 7]
 [12]
 [ 4]
 [ 5]
 [11]
 [ 1]
 [ 8]
 [10]
 [ 3]
 [ 6]
 [ 9]]'''

def get_pattern_file(pattern_name):
    """
    Return the full path to a pattern file based on the pattern name.
    
    Args:
        pattern_name: Name of the pattern
        
    Returns:
        Full path to the pattern file
    """
    # Pattern files location
    pattern_files = {
        "1296x64_rowbar_4states": "State_pattern_files/1296x64_rowbar_4states.npy",
        "3x4_4states_debug": "State_pattern_files/3x4_4states_debug.npy",
        "248x248_checkerboard_4states": "State_pattern_files/248x248_checkerboard_4states.npy",
        "1296x64_Adrien_random_4states": "State_pattern_files/1296x64_Adrien_random_4states.npy",
        "248x248_1state": "State_pattern_files/248x248_1state.npy",
        "1296x64_1state": "State_pattern_files/1296x64_1state.npy",
        "248x248_16states": "State_pattern_files/248x248_16states.npy",
        "248x248_2states": "State_pattern_files/248x248_2states.npy",
        "62x62_2states": "State_pattern_files/62x62_2states.npy",
        "248x1_1state": "State_pattern_files/248x1_1state.npy",
        "82944x78_ecc_fuxi": "State_pattern_files/82944x78_ecc_fuxi.npy",
        "65536x78_ecc": "State_pattern_files/65536x78_ecc.npy",
        "248x256_1state": "State_pattern_files/248x256_1state.npy",
        "256x32_pr0": "State_pattern_files/256x32_pr0.npy",
        "256x32_pr1": "State_pattern_files/256x32_pr1.npy",
    }

    '''pattern_files = {
        "1296x64_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_rowbar_4states.npy",
        "3x4_4states_debug": "/home/admin2/webapp_2/State_pattern_files/3x4_4states_debug.npy",
        "248x248_checkerboard_4states": "/home/admin2/webapp_2/State_pattern_files/248x248_checkerboard_4states.npy",
        "1296x64_Adrien_random_4states": "State_pattern_files/1296x64_Adrien_random_4states.npy",
        "248x248_1state": "/home/admin2/webapp_2/State_pattern_files/248x248_1state.npy",
        "1296x64_1state": "/home/admin2/webapp_2/State_pattern_files/1296x64_1state.npy",
        "248x248_16states": "/home/admin2/webapp_2/State_pattern_files/248x248_16states.npy",
        "248x248_2states": "/home/admin2/webapp_2/State_pattern_files/248x248_2states.npy",
        "62x62_2states": "/home/admin2/webapp_2/State_pattern_files/62x62_2states.npy",
        "248x1_1state": "/home/admin2/webapp_2/State_pattern_files/248x1_1state.npy",
        "82944x78_ecc_fuxi": "/home/admin2/webapp_2/State_pattern_files/82944x78_ecc_fuxi.npy",
        "65536x78_ecc": "/home/admin2/webapp_2/State_pattern_files/65536x78_ecc.npy",
        "248x256_1state": "/home/admin2/webapp_2/State_pattern_files/248x256_1state.npy",
        "256x32_pr0": "/home/admin2/webapp_2/State_pattern_files/256x32_pr0.npy",
        "256x32_pr1": "/home/admin2/webapp_2/State_pattern_files/256x32_pr1.npy",
    }'''
    
    # Return the path for the pattern name
    return pattern_files.get(pattern_name, "")

@app.route('/mergeTablesProcess', methods=['POST'])
def merge_tables_process():
    database = request.form.get('database')
    table_names = request.form.getlist('tableNames')
    state_pattern = request.form.get('state_pattern')
    new_table_name = request.form.get('newTableName')

    print(f"DEBUG: Starting merge process with parameters:")
    print(f"DEBUG: Database: {database}")
    print(f"DEBUG: Table names: {table_names}")
    print(f"DEBUG: State pattern: {state_pattern}")
    print(f"DEBUG: New table name: {new_table_name}")

    if not database:
        return 'Database not specified', 400
    if not table_names:
        return 'No tables specified', 400
    if not state_pattern:
        return 'State pattern not specified', 400
    if not new_table_name:
        return 'New table name not specified', 400

    # Import packages needed for the function
    import numpy as np
    import pandas as pd
    import traceback
    from db_operations import create_long_running_connection, create_long_running_engine

    try:
        # Use a long-running connection
        connection = create_long_running_connection(database)
        cursor = connection.cursor()

        # Configure MySQL session for handling wide tables
        cursor.execute("SET SESSION innodb_strict_mode=OFF")
        cursor.execute("SET SESSION sql_mode=''")
        # Removed: # Removed: cursor.execute("SET GLOBAL innodb_file_per_table=ON")
        # Add more optimization settings for extremely wide tables
        # Removed: cursor.execute("SET SESSION innodb_fill_factor=70")
        # Removed: cursor.execute("SET SESSION max_allowed_packet=1073741824")  # Set to 1GB
        cursor.execute("SET SESSION optimizer_switch='mrr=on,mrr_cost_based=off'")
        connection.commit()
        
        # First, get a count of the total columns across all tables
        total_columns = 0
        for table_name in table_names:
            try:
                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
                columns = cursor.fetchall()
                total_columns += len(columns)
            except Exception as e:
                print(f"DEBUG: Error getting columns from {table_name}: {str(e)}")
                continue
        
        print(f"DEBUG: Processing {len(table_names)} tables with a total of {total_columns} columns")
        if total_columns > 500:
            print(f"DEBUG: Warning - Very large number of columns ({total_columns}). This might exceed MySQL row size limits.")
            print("DEBUG: The tables may be split into multiple tables.")

        # Load pattern from the predefined JSON sets
        pattern_source = state_pattern
        
        # Load the pattern array from the pattern storage directory
        pattern_file = get_pattern_file(pattern_source)
        print(f"DEBUG: Loading pattern from file: {pattern_file}")
        
        # Check if the pattern file exists
        if not os.path.exists(pattern_file):
            print(f"DEBUG: Pattern file not found: {pattern_file}")
            return f'Pattern file not found: {pattern_file}', 400
        
        # Load the pattern array
        pattern_array = np.load(pattern_file, allow_pickle=True)
        
        # Special handling for 82944x78_ecc_fuxi.npy which is actually (78, 1296, 64)
        if state_pattern == "82944x78_ecc_fuxi":
            # Reshape the 3D array to 2D (78, 82944) and then transpose to (82944, 78)
            print(f"DEBUG: Special handling for 82944x78_ecc_fuxi pattern")
            if len(pattern_array.shape) == 3:
                pattern_array = pattern_array.reshape(pattern_array.shape[0], pattern_array.shape[1] * pattern_array.shape[2]).T
            print(f"DEBUG: After reshaping, pattern array shape: {pattern_array.shape}")
        
        # Special handling for 65536x78_ecc.npy which is actually (78, 32, 2048)
        if state_pattern == "65536x78_ecc":
            # Reshape the 3D array to 2D (78, 65536) and then transpose to (65536, 78)
            print(f"DEBUG: Special handling for 65536x78_ecc pattern")
            if len(pattern_array.shape) == 3:
                pattern_array = pattern_array.reshape(pattern_array.shape[0], pattern_array.shape[1] * pattern_array.shape[2]).T
            print(f"DEBUG: After reshaping, pattern array shape: {pattern_array.shape}")
        
        print(f"DEBUG: Pattern array shape: {pattern_array.shape}, dtype: {pattern_array.dtype}")
        
        a, b = pattern_array.shape
        print(f"DEBUG: Pattern dimensions: a={a}, b={b}")
        
        pattern_flat = pattern_array.flatten()
        print(f"DEBUG: Pattern flat shape: {pattern_flat.shape}, size: {pattern_flat.size}")
        print(f"DEBUG: Pattern flat min: {pattern_flat.min()}, max: {pattern_flat.max()}")
        
        # Print the first few elements of pattern_flat to verify content
        print(f"DEBUG: First 10 elements of pattern_flat: {pattern_flat[:10]}")
        
        # Count unique values in pattern
        unique_values, counts = np.unique(pattern_flat, return_counts=True)
        print(f"DEBUG: Unique values in pattern: {unique_values}")
        print(f"DEBUG: Counts of unique values: {counts}")

        # Process tables in batches to avoid memory issues
        BATCH_SIZE = 50  # Process 50 tables at a time
        processed_tables = 0
        total_tables = len(table_names)
        
        # Initialize reshaped_arrays list
        reshaped_arrays = []
        
        # Process tables in batches
        for batch_start in range(0, total_tables, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_tables)
            current_batch = table_names[batch_start:batch_end]
            
            print(f"\nDEBUG: Processing batch of tables: {batch_start+1} to {batch_end} of {total_tables}")
            
            # Process each table in the current batch
            for table_name in current_batch:
                print(f"\nDEBUG: Processing table: {table_name}")
                
                # Create a new connection for each table to avoid timeouts
                with create_long_running_connection(database) as batch_conn:
                    batch_cursor = batch_conn.cursor()
                    
                    query = f"SELECT * FROM `{table_name}`"
                    batch_cursor.execute(query)
                    rows = batch_cursor.fetchall()
                    print(f"DEBUG: Fetched {len(rows)} rows from table")
                    
                    if len(rows) == 0:
                        print(f"DEBUG: Warning - Empty table: {table_name}")
                        continue
                        
                    columns = [desc[0] for desc in batch_cursor.description]
                    print(f"DEBUG: Column count: {len(columns)}")
                
                # Process data outside the connection context
                df = pd.DataFrame(rows, columns=columns)
                print(f"DEBUG: DataFrame shape: {df.shape}")
                
                # Check for missing values
                if df.isnull().values.any():
                    print("DEBUG: Warning - DataFrame contains NaN values")
                
                # Convert DataFrame to numpy array
                arr = df.to_numpy()
                print(f"DEBUG: Array shape: {arr.shape}, dtype: {arr.dtype}")
                
                # Check if the array is compatible with the pattern
                if arr.shape[0] % a != 0 or arr.shape[1] % b != 0:
                    print(f"DEBUG: Warning - Array dimensions {arr.shape} not divisible by pattern dimensions {pattern_array.shape}")
                    continue
                
                # Reshape and apply the pattern - this is the key operation
                try:
                    # Calculate the number of repetitions needed
                    m = arr.shape[0] // a  # Number of repeats in the row dimension
                    n = arr.shape[1] // b  # Number of repeats in the column dimension
                    print(f"DEBUG: Reshaping with m={m}, n={n}")
                    
                    # Reshape to (m, a, n, b) for proper tiling and pattern application
                    arr_4d = arr.reshape(m, a, n, b)
                    print(f"DEBUG: 4D array shape: {arr_4d.shape}")
                    
                    # Transpose to group the pattern dimensions together
                    arr_4d = arr_4d.transpose(0, 2, 1, 3)
                    print(f"DEBUG: 4D transposed shape: {arr_4d.shape}")
                    
                    # Reshape to (m*n, a*b)
                    arr_2d = arr_4d.reshape(m*n, a*b)
                    print(f"DEBUG: 2D array shape: {arr_2d.shape}")
                    
                    # Flatten pattern and get indices that would sort it
                    pattern_idx = np.argsort(pattern_flat)
                    print(f"DEBUG: Pattern indices shape: {pattern_idx.shape}")
                    
                    # Reorder the array columns based on pattern
                    reordered = arr_2d[:, pattern_idx]
                    print(f"DEBUG: Reordered array shape: {reordered.shape}")
                    
                    # Reshape to (m*n*a*b, 1)
                    reshaped = reordered.reshape(m*n*a*b, 1)
                    print(f"DEBUG: Final reshaped array shape: {reshaped.shape}")
                    
                    # Add to reshaped_arrays list
                    reshaped_arrays.append(reshaped)
                    
                    processed_tables += 1
                    print(f"DEBUG: Successfully processed table {table_name}")
                    
                except Exception as e:
                    print(f"DEBUG: Error reshaping table {table_name}: {str(e)}")
                    continue
        
        if reshaped_arrays:
            # Concatenate all reshaped arrays along axis=1
            print("DEBUG: Concatenating arrays...")
            try:
                combined_array = np.concatenate(reshaped_arrays, axis=1)
                print(f"DEBUG: Combined array shape: {combined_array.shape}")
            except Exception as e:
                print(f"DEBUG: Error during concatenation: {str(e)}")
                return f'Error during concatenation: {str(e)}', 500

            # Convert back to DataFrame
            print("DEBUG: Converting to DataFrame...")
            combined_df = pd.DataFrame(combined_array)
            print(f"DEBUG: Combined DataFrame shape: {combined_df.shape}")

            # Rename duplicate columns
            print("DEBUG: Renaming duplicate columns...")
            combined_df = rename_duplicate_columns(combined_df)

            # Check if the new table name already exists
            cursor.execute("SHOW TABLES LIKE %s", (new_table_name,))
            if cursor.fetchone():
                connection.close()
                return 'A table with the new name already exists.', 400

            # Create the new table in the database with chunked insertion
            print(f"DEBUG: Saving data to new table: {new_table_name}")
            
            # Create long-running engine
            engine = create_long_running_engine(database)
            
            # Calculate how many tables we need to split this into
            # MySQL has row size limitations, so we'll split into multiple tables if needed
            # Using a reasonable limit to balance between row size and table count
            MAX_COLUMNS_PER_TABLE = 500  # Changed back to 500 from 1000
            
            # Allow user to force a single table regardless of column count
            force_single_table = True  # Set to True to always create a single table
            
            total_columns = len(combined_df.columns)
            if force_single_table:
                num_tables_needed = 1
                print(f"DEBUG: Forcing single table mode, keeping all {total_columns} columns in one table")
            else:
                num_tables_needed = (total_columns + MAX_COLUMNS_PER_TABLE - 1) // MAX_COLUMNS_PER_TABLE
                print(f"DEBUG: Total columns: {total_columns}, splitting into {num_tables_needed} tables")
            
            # Track all created table names for success message
            created_table_names = []
            
            # Split the columns into groups and create multiple tables
            for table_idx in range(num_tables_needed):
                start_col = table_idx * MAX_COLUMNS_PER_TABLE
                end_col = min((table_idx + 1) * MAX_COLUMNS_PER_TABLE, total_columns)
                
                # Create a name for this part table
                if num_tables_needed > 1:
                    part_table_name = f"{new_table_name}_{table_idx + 1}"
                else:
                    part_table_name = new_table_name
                
                # Get the columns for this table
                table_columns = combined_df.columns[start_col:end_col].tolist()
                print(f"DEBUG: Creating table {part_table_name} with columns {start_col} to {end_col-1}")
                
                try:
                    # Drop the table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS `{part_table_name}`")
                    
                    # Set MySQL optimization settings to help with wide tables
                    cursor.execute("SET SESSION innodb_strict_mode=OFF")
                    cursor.execute("SET SESSION sql_mode=''")
                    # Removed: cursor.execute("SET GLOBAL innodb_file_per_table=ON")
                    
                    # Create column definitions for this table
                    column_defs = []
                    for col_name in table_columns:
                        # Use TEXT instead of VARCHAR(255) to avoid row size limits
                        # TEXT data types are stored separately and only pointers are kept in the row
                        column_defs.append(f"`{col_name}` TEXT")
                    
                    # Create the table with DYNAMIC row format
                    create_table_sql = f"""
                    CREATE TABLE `{part_table_name}` (
                        {', '.join(column_defs)}
                    ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC
                    """
                    
                    print(f"DEBUG: Creating table {part_table_name} with SQL: {create_table_sql}")
                    cursor.execute(create_table_sql)
                    connection.commit()
                    
                    # Track the created table
                    created_table_names.append(part_table_name)
                    
                    # Now insert data in chunks
                    CHUNK_SIZE = 1000  # Use smaller chunks for insertion
                    total_rows = len(combined_df)
                    
                    # Process data in smaller batches to avoid timeouts
                    for i in range(0, total_rows, CHUNK_SIZE):
                        chunk = combined_df.iloc[i:i + CHUNK_SIZE, start_col:end_col]
                        # Prepare column names and placeholders for SQL
                        columns_str = ", ".join([f"`{col}`" for col in table_columns])
                        placeholders = ", ".join(["%s"] * len(table_columns))
                        
                        # Create INSERT statement - directly reference columns without ID field
                        insert_sql = f"INSERT INTO `{part_table_name}` ({columns_str}) VALUES ({placeholders})"
                        
                        # Convert rows to list of tuples for executemany
                        values = []
                        for _, row in chunk.iterrows():
                            # Convert any NaN values to NULL for MySQL
                            row_values = []
                            for col in table_columns:
                                val = row[col]
                                if pd.isna(val):
                                    row_values.append(None)
                                else:
                                    row_values.append(str(val))
                            values.append(tuple(row_values))
                        
                        # Execute batch insert
                        cursor.executemany(insert_sql, values)
                        
                        # Commit after each chunk
                        connection.commit()
                        print(f"DEBUG: Table {part_table_name}: Inserted chunk {i//CHUNK_SIZE + 1} of {(total_rows + CHUNK_SIZE - 1)//CHUNK_SIZE}")
                    
                    print(f"DEBUG: Table {part_table_name} created and populated successfully")
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"DEBUG: Error creating table {part_table_name}: {error_message}")
                    print(f"DEBUG: Traceback: {traceback.format_exc()}")
                    
                    # Provide a more helpful error message for row size issues
                    if "row size too large" in error_message.lower():
                        return (f'Error: Row size too large for table {part_table_name}. '
                               f'The table has {len(table_columns)} columns which exceeds MySQL limits. '
                               f'Try turning off "force_single_table" to split into multiple tables.'), 500
                    
                    return f'Error creating table {part_table_name}: {error_message}', 500
            
            # Success message based on how many tables were created
            if num_tables_needed > 1:
                success_message = f"Data was split into {num_tables_needed} tables due to column limits: {', '.join(created_table_names)}"
                print(f"DEBUG: {success_message}")
                flash(success_message, 'info')
            else:
                print("DEBUG: Table saved successfully")
                
            # Clean up
            cursor.close()
            connection.close()

            # After successful merging, redirect to list_tables
            return redirect(url_for('list_tables', database=database))

        else:
            # Clean up
            cursor.close()
            connection.close()
            return 'No tables were reshaped and combined.', 400

    except Exception as e:
        print(f"DEBUG: Unexpected error: {str(e)}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return str(e), 500

@app.route('/copy_tables', methods=['POST'])
def copy_tables():
    data = request.get_json()
    source_db = data.get('sourceDatabase')
    target_db = data.get('targetDatabase')
    table_names = data.get('tableNames')

    # Check for missing data
    if not all([source_db, target_db, table_names]):
        return jsonify({'message': 'Missing data in request.'}), 400

    # Connect to databases - use long running connections for large tables
    try:
        from db_operations import create_long_running_connection, get_table_names
        source_conn = create_long_running_connection(source_db)
        target_conn = create_long_running_connection(target_db)
        
        # Get existing table names in the target database
        existing_tables = get_table_names(target_conn)

        # Find conflicts
        conflicts = set(table_names) & set(existing_tables)
        if conflicts:
            conflict_list = ', '.join(conflicts)
            source_conn.close()
            target_conn.close()
            return jsonify({'message': f'The following tables already exist in the target database: {conflict_list}'}), 400

        # Copy tables one by one
        for table in table_names:
            print(f"Starting to copy table: {table}")
            
            # Get table structure first
            source_cursor = source_conn.cursor()
            try:
                # Get the column information
                source_cursor.execute(f"DESCRIBE `{table}`")
                columns_info = source_cursor.fetchall()
                column_names = [col[0] for col in columns_info]
                column_types = [col[1] for col in columns_info]
                
                # Count total columns
                num_columns = len(column_names)
                print(f"Table {table} has {num_columns} columns")
                
                # Check row count for chunking
                source_cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                row_count = source_cursor.fetchone()[0]
                print(f"Table {table} has {row_count} rows")
                
                # Create target table with the same structure
                target_cursor = target_conn.cursor()
                
                # Flag to track if we need to resort to vertical partitioning
                use_vertical_partitioning = False
                
                # First attempt: Try to create an exact copy with original structure
                try:
                    # Try to create the table with the exact same structure as the source
                    source_cursor.execute(f"SHOW CREATE TABLE `{table}`")
                    create_table_sql = source_cursor.fetchone()[1]
                    
                    # Remove AUTO_INCREMENT values and other potential incompatibilities
                    create_table_sql = re.sub(r'AUTO_INCREMENT=\d+', '', create_table_sql)
                    
                    print(f"Attempting to create table with original structure")
                    target_cursor.execute(create_table_sql)
                    target_conn.commit()
                    print(f"Successfully created table with original structure")
                    
                    # Use exact same structure for copying
                    exact_copy = True
                except Exception as create_error:
                    print(f"Could not create exact copy: {create_error}")
                    exact_copy = False
                
                # Second attempt: Try with all TEXT columns if first attempt failed
                if not exact_copy:
                    try:
                        # Use InnoDB with DYNAMIC row format to handle wide rows better
                        table_options = "ENGINE=InnoDB ROW_FORMAT=DYNAMIC"
                        
                        # For wide tables, use more aggressive column type conversion
                        column_defs = []
                        for i, col_name in enumerate(column_names):
                            # For tables with many columns, convert all string types to TEXT
                            # and reduce size of numeric columns
                            col_type = column_types[i].upper()
                            
                            # Convert string types to TEXT to reduce row overhead
                            if ('CHAR' in col_type or 'TEXT' in col_type or 'BLOB' in col_type):
                                col_type = 'TEXT'
                            # Reduce size of INT columns in wide tables
                            elif 'INT' in col_type and num_columns > 100:
                                # Use smaller integer types for extremely wide tables
                                col_type = 'INT'
                            # Keep other types as is
                            
                            column_defs.append(f"`{col_name}` {col_type}")
                        
                        # Create the table with optimized structure
                        create_table_sql = f"CREATE TABLE `{table}` ({', '.join(column_defs)}) {table_options}"
                        print(f"Attempting to create table with optimized structure")
                        target_cursor.execute(create_table_sql)
                        target_conn.commit()
                        
                        # Try to further optimize with additional settings
                        try:
                            # Additional optimizations for wide tables
                            target_cursor.execute(f"ALTER TABLE `{table}` ROW_FORMAT=DYNAMIC")
                            target_conn.commit()
                            print(f"Set ROW_FORMAT=DYNAMIC")
                        except Exception as format_error:
                            print(f"Could not set row format: {format_error}")
                            # Continue anyway
                        
                        print(f"Successfully created table with optimized structure")
                    except Exception as opt_error:
                        print(f"Could not create optimized structure: {opt_error}")
                        
                        # Third attempt: Try with all TEXT and minimal options
                        try:
                            print(f"Attempting with all TEXT columns and minimal options")
                            column_defs = [f"`{col}` TEXT" for col in column_names]
                            simple_sql = f"CREATE TABLE `{table}` ({', '.join(column_defs)})"
                            
                            target_cursor.execute(simple_sql)
                            target_conn.commit()
                            print(f"Created table with all TEXT columns")
                        except Exception as text_error:
                            print(f"Could not create even with all TEXT: {text_error}")
                            use_vertical_partitioning = True
                
                # If we're not using vertical partitioning yet, try to copy the data
                if not use_vertical_partitioning:
                    try:
                        # Process data in batches and chunks for large tables
                        BATCH_SIZE = 1000
                        
                        # For regular tables, process in simple batches
                        source_cursor.execute(f"SELECT * FROM `{table}`")
                        data = source_cursor.fetchall()

                        # Insert data in batches
                        if data:
                            placeholders = ", ".join(["%s"] * len(column_names))
                            insert_sql = f"INSERT INTO `{table}` ({', '.join([f'`{col}`' for col in column_names])}) VALUES ({placeholders})"

                            for i in range(0, len(data), BATCH_SIZE):
                                batch = data[i:i+BATCH_SIZE]
                                target_cursor.executemany(insert_sql, batch)
                                target_conn.commit()
                                print(f"Inserted batch {i//BATCH_SIZE + 1} of {(len(data) + BATCH_SIZE - 1)//BATCH_SIZE}")

                            print(f"Successfully copied table {table}")
                    except Exception as copy_error:
                        print(f"Error copying data: {copy_error}")
                        if "Row size too large" in str(copy_error):
                            print("Row size error detected. Trying vertical partitioning as a last resort.")
                            use_vertical_partitioning = True
                        else:
                            raise
                
                # Fall back to vertical partitioning if all other attempts failed
                if use_vertical_partitioning:
                    print(f"Using vertical partitioning as a last resort for {table}")
                    
                    # Clean up any partially created table
                    try:
                        target_cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
                        target_conn.commit()
                    except Exception as drop_error:
                        print(f"Error dropping table: {drop_error}")
                    
                    # Split the table into multiple tables, each with at most 40 columns per table
                    MAX_COLUMNS_PER_TABLE = 40
                    
                    # Get the total number of parts we'll need
                    total_parts = (num_columns // MAX_COLUMNS_PER_TABLE) + (1 if num_columns % MAX_COLUMNS_PER_TABLE > 0 else 0)
                    print(f"Splitting into {total_parts} tables")
                    
                    # Generate a unique ID column that will be added to all tables for joining
                    id_col_name = 'row_id_for_joining'
                    
                    # Process the table in chunks
                    for part_index in range(total_parts):
                        start_col = part_index * MAX_COLUMNS_PER_TABLE
                        end_col = min((part_index + 1) * MAX_COLUMNS_PER_TABLE, num_columns)
                        
                        # Create a unique name for this part
                        part_table_name = f"_{table}_part{part_index+1}"  # Add underscore prefix to hide tables
                        
                        # Check if this part table already exists
                        target_cursor.execute(f"SHOW TABLES LIKE '{part_table_name}'")
                        if target_cursor.fetchone():
                            # If it exists, we need to drop it first
                            target_cursor.execute(f"DROP TABLE `{part_table_name}`")
                            target_conn.commit()
                            
                        # Generate column definitions for this part
                        part_column_defs = [f"`{id_col_name}` INT NOT NULL PRIMARY KEY"]  # Add an ID column
                        
                        # Add the columns for this part
                        part_columns = column_names[start_col:end_col]
                        for i, col_name in enumerate(part_columns):
                            col_index = start_col + i
                            col_type = "TEXT"  # Just use TEXT for everything to avoid type issues
                            part_column_defs.append(f"`{col_name}` {col_type}")
                        
                        # Create the part table
                        create_sql = f"CREATE TABLE `{part_table_name}` ({', '.join(part_column_defs)})"
                        print(f"Creating part table: {create_sql}")
                        target_cursor.execute(create_sql)
                        target_conn.commit()
                        
                        print(f"Created part table {part_table_name} with columns {start_col} to {end_col-1}")
                        
                        # Copy data to this part table in batches
                        BATCH_SIZE = 1000
                        for offset in range(0, row_count, BATCH_SIZE):
                            limit = min(BATCH_SIZE, row_count - offset)
                            
                            # First, retrieve the ID column if it's the first part (otherwise we'll use the already generated IDs)
                            if part_index == 0:
                                # For the first part, we need to generate and store row IDs
                                
                                # Build SQL to select columns for this part
                                select_cols = ", ".join([f"`{col}`" for col in part_columns])
                                source_cursor.execute(f"SELECT {select_cols} FROM `{table}` LIMIT {limit} OFFSET {offset}")
                                batch_data = source_cursor.fetchall()
                                
                                if batch_data:
                                    # Insert into the part table with generated IDs
                                    insert_cols = [id_col_name] + part_columns
                                    insert_cols_sql = ", ".join([f"`{col}`" for col in insert_cols])
                                    
                                    # Create row IDs for this batch (start from offset+1)
                                    rows_with_ids = []
                                    for row_idx, row in enumerate(batch_data):
                                        rows_with_ids.append((offset + row_idx + 1,) + row)  # Add ID at the beginning
                                    
                                    # Insert with IDs
                                    placeholders = ", ".join(["%s"] * len(insert_cols))
                                    insert_sql = f"INSERT INTO `{part_table_name}` ({insert_cols_sql}) VALUES ({placeholders})"
                                    
                                    target_cursor.executemany(insert_sql, rows_with_ids)
                                    target_conn.commit()
                                    
                                    print(f"Inserted {len(batch_data)} rows into part table {part_table_name} (with IDs)")
                            else:
                                # For subsequent parts, we need to match with the already created IDs
                                
                                # First get the existing row IDs
                                first_part_name = f"_{table}_part1"  # Using updated name with underscore prefix
                                target_cursor.execute(f"SELECT `{id_col_name}` FROM `{first_part_name}` ORDER BY `{id_col_name}` LIMIT {limit} OFFSET {offset}")
                                existing_ids = [row[0] for row in target_cursor.fetchall()]
                                
                                if existing_ids:
                                    # Now get the corresponding data from the source
                                    select_cols = ", ".join([f"`{col}`" for col in part_columns])
                                    source_cursor.execute(f"SELECT {select_cols} FROM `{table}` LIMIT {limit} OFFSET {offset}")
                                    batch_data = source_cursor.fetchall()
                                    
                                    if batch_data:
                                        # Combine IDs with data
                                        rows_with_ids = []
                                        for id_val, row in zip(existing_ids, batch_data):
                                            rows_with_ids.append((id_val,) + row)  # Add ID at the beginning
                                        
                                        # Insert with existing IDs
                                        insert_cols = [id_col_name] + part_columns
                                        insert_cols_sql = ", ".join([f"`{col}`" for col in insert_cols])
                                        placeholders = ", ".join(["%s"] * len(insert_cols))
                                        insert_sql = f"INSERT INTO `{part_table_name}` ({insert_cols_sql}) VALUES ({placeholders})"
                                        
                                        target_cursor.executemany(insert_sql, rows_with_ids)
                                        target_conn.commit()
                                        
                                        print(f"Inserted {len(batch_data)} rows into part table {part_table_name} (with existing IDs)")
                    
                    # Create a view that joins all the parts to make it look like a single table
                    try:
                        view_name = f"{table}"  # Use the original table name for the view
                        
                        # Check if view exists and drop it
                        target_cursor.execute(f"SHOW TABLES LIKE '{view_name}'")
                        if target_cursor.fetchone():
                            target_cursor.execute(f"DROP VIEW `{view_name}`")
                            target_conn.commit()
                        
                        # Build view SQL
                        view_sql = f"CREATE VIEW `{view_name}` AS SELECT "
                        
                        # Add all columns from all parts
                        all_column_selects = []
                        
                        for part_index in range(total_parts):
                            part_name = f"p{part_index+1}"
                            start_col = part_index * MAX_COLUMNS_PER_TABLE
                            end_col = min((part_index + 1) * MAX_COLUMNS_PER_TABLE, num_columns)
                            
                            # Get the columns for this part
                            part_columns = column_names[start_col:end_col]
                            for col in part_columns:
                                all_column_selects.append(f"{part_name}.`{col}`")
                        
                        # Add all columns to the view
                        view_sql += ", ".join(all_column_selects)
                        
                        # Add FROM clause
                        view_sql += f" FROM `_{table}_part1` p1"
                        
                        # Add JOIN clauses for all other parts
                        for part_index in range(1, total_parts):
                            part_name = f"p{part_index+1}"
                            part_table = f"_{table}_part{part_index+1}"  # Using updated name with underscore prefix
                            view_sql += f" JOIN `{part_table}` {part_name} ON p1.`{id_col_name}` = {part_name}.`{id_col_name}`"
                        
                        # Create the view
                        target_cursor.execute(view_sql)
                        target_conn.commit()
                        print(f"Created view '{view_name}' that looks like the original table by joining all {total_parts} parts")
                        
                        # Success message mentioning the special case
                        print(f"Successfully copied wide table {table} using vertical partitioning with a unified view")
                        
                        # NEW CONSOLIDATION STEP: Extract data from view and create a single table
                        try:
                            print(f"Starting consolidation: Converting partitioned view into a single physical table")
                            
                            # First, create a new table with the _single suffix as a temporary name
                            consolidated_temp_name = f"{table}_single_temp"
                            
                            # Drop the temp table if it exists
                            target_cursor.execute(f"DROP TABLE IF EXISTS `{consolidated_temp_name}`")
                            target_conn.commit()
                            
                            # Get column definitions for the consolidated table (all TEXT to be safe)
                            col_defs = [f"`{col}` TEXT" for col in column_names]
                            create_consolidated_sql = f"CREATE TABLE `{consolidated_temp_name}` ({', '.join(col_defs)})"
                            
                            # Create the consolidated table
                            target_cursor.execute(create_consolidated_sql)
                            target_conn.commit()
                            print(f"Created consolidated temporary table {consolidated_temp_name}")
                            
                            # Insert data from the view into the consolidated table
                            # Use a low batch size for large tables
                            SAFE_BATCH_SIZE = 500
                            
                            # Get the total number of rows
                            view_count = row_count  # We already know this
                            
                            # Process in chunks to avoid memory issues
                            for offset in range(0, view_count, SAFE_BATCH_SIZE):
                                # Only select the actual columns (not the row_id used for joining)
                                cols_sql = ", ".join([f"`{col}`" for col in column_names])
                                
                                # Get data from view
                                target_cursor.execute(f"SELECT {cols_sql} FROM `{view_name}` LIMIT {SAFE_BATCH_SIZE} OFFSET {offset}")
                                view_data = target_cursor.fetchall()
                                
                                if view_data:
                                    # Insert into consolidated table
                                    placeholders = ", ".join(["%s"] * len(column_names))
                                    insert_sql = f"INSERT INTO `{consolidated_temp_name}` ({cols_sql}) VALUES ({placeholders})"
                                    
                                    target_cursor.executemany(insert_sql, view_data)
                                    target_conn.commit()
                                    print(f"Inserted batch {offset//SAFE_BATCH_SIZE + 1} of {(view_count + SAFE_BATCH_SIZE - 1)//SAFE_BATCH_SIZE} into consolidated table")
                            
                            # Drop the view
                            target_cursor.execute(f"DROP VIEW `{view_name}`")
                            target_conn.commit()
                            print(f"Dropped view {view_name}")
                            
                            # Drop all the part tables
                            for i in range(1, total_parts + 1):
                                part_name = f"_{table}_part{i}"
                                target_cursor.execute(f"DROP TABLE IF EXISTS `{part_name}`")
                                target_conn.commit()
                                print(f"Dropped part table {part_name}")
                            
                            # Rename the consolidated table to the final name
                            target_cursor.execute(f"RENAME TABLE `{consolidated_temp_name}` TO `{table}`")
                            target_conn.commit()
                            print(f"Renamed consolidated table to {table}")
                            
                            print(f"Consolidation complete: Segmented tables have been replaced with a single physical table")
                            
                        except Exception as consolidation_error:
                            print(f"Warning: Could not consolidate into a single table: {consolidation_error}")
                            print(f"The view and segment tables will remain as they are.")
                    except Exception as view_error:
                        print(f"Warning: Could not create the unified view: {view_error}")
                        print(f"Table is split into multiple parts. Use the part tables directly.")
            
            except Exception as e:
                print(f"Error copying table {table}: {e}")
                import traceback
                traceback.print_exc()
                source_conn.close()
                target_conn.close()
                return jsonify({'message': f'Error copying table {table}: {str(e)}'}), 500

    except Exception as e:
        print(f'Error copying tables: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'message': f'An error occurred while copying tables: {str(e)}'}), 500

    finally:
        if 'source_cursor' in locals() and source_cursor:
            source_cursor.close()
        if 'target_cursor' in locals() and target_cursor:
            target_cursor.close()

        # Close database connections
        source_conn.close()
        target_conn.close()
        
    # Return success message at the end of try block (after the for loop)
    return jsonify({'message': 'Tables copied successfully.'}), 200

@app.route('/concatenate_tables', methods=['POST'])
def concatenate_tables():
    data = request.json
    database = data.get('database')
    table_names = data.get('tableNames')
    new_table_name = data.get('newTableName')
    
    # Add a force_single_table parameter set to True to ensure we always create one table
    force_single_table = True
    
    # Set a reasonable batch size for processing large tables
    BATCH_SIZE = 5  # Process 5 tables at a time
    # MODIFIED: Increase MAX_COLUMNS_PER_BATCH to avoid splitting tables
    MAX_COLUMNS_PER_BATCH = 2000  # Increased from 1000 to 2000
    
    # MySQL has a limit of 1024 columns per table in older versions, 4096 in newer
    MYSQL_COLUMN_LIMIT = 4096  # Updated to newer MySQL limit

    # Import packages needed for the function
    import numpy as np
    import pandas as pd
    import traceback

    if not (database and table_names and new_table_name):
        return jsonify(success=False, message='Missing required information.')

    # MySQL has a table name limit of 64 characters
    MAX_TABLE_NAME_LENGTH = 64
    
    # Check if the new table name is too long for MySQL
    if len(new_table_name) > MAX_TABLE_NAME_LENGTH:
        return jsonify(
            success=False, 
            message=f'Table name "{new_table_name}" is too long. MySQL has a limit of {MAX_TABLE_NAME_LENGTH} characters.',
            skipped=True
        )

    try:
        # Use our new long-running connection instead of the regular one
        from db_operations import create_long_running_connection, create_long_running_engine
        connection = create_long_running_connection(database)
        cursor = connection.cursor()

        # Set MySQL optimization settings to help with wide tables
        cursor.execute("SET SESSION innodb_strict_mode=OFF")
        cursor.execute("SET SESSION sql_mode=''")
        # Removed: cursor.execute("SET SESSION innodb_fill_factor=70")
        # Removed: cursor.execute("SET SESSION max_allowed_packet=1073741824")  # 1GB
        # Removed: cursor.execute("SET SESSION innodb_large_prefix=ON")
        # Removed: cursor.execute("SET GLOBAL innodb_file_per_table=ON")
        connection.commit()

        # First, check if all tables have the same row count
        row_counts = {}
        for table_name in table_names:
            try:
                # Just get the count, not all data
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                row_counts[table_name] = count
            except Exception as e:
                print(f"Error getting row count for table {table_name}: {e}")
                connection.close()
                return jsonify(success=False, message=f'Error accessing table {table_name}: {str(e)}')
        
        # If there are no tables with rows, we can't proceed
        if not row_counts:
            connection.close()
            return jsonify(success=False, message='No tables with data to concatenate.')
        
        # Check that all tables have the same count
        if len(set(row_counts.values())) != 1:
            connection.close()
            return jsonify(success=False, 
                           message='Tables have different row counts. All tables must have the same number of rows.',
                           row_counts=row_counts)
        
        # Get the row count for processing
        row_count = next(iter(row_counts.values()))
        
        # Check if the new table name already exists - do this early before processing
        cursor.execute("SHOW TABLES LIKE %s", (new_table_name,))
        if cursor.fetchone():
            connection.close()
            return jsonify(success=False, message='A table with that name already exists.')
        
        # Get the total number of columns across all tables to concatenate
        all_column_info = []
        total_columns = 0
        
        for i, table_name in enumerate(table_names):
            try:
                cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
                columns = [col[0] for col in cursor.fetchall()]
                all_column_info.append({
                    'table_name': table_name,
                    'table_index': i + 1,  # 1-based index for clearer naming
                    'columns': columns
                })
                total_columns += len(columns)
            except Exception as e:
                print(f"Error getting columns for table {table_name}: {e}")
                connection.close()
                return jsonify(success=False, message=f'Error getting columns for table {table_name}: {str(e)}')
        
        print(f"Total columns to concatenate: {total_columns}")
        
        # We're now bypassing the split approach to create a single table
        # This was the previous condition:
        # if total_columns > 200 or force_split:
        if False:  # Never trigger the split approach
            print(f"Wide table detected ({total_columns} columns). Will use split table approach.")
            try:
                # Create multiple tables, each with a portion of the columns
                max_columns_per_table = 50  # Limit to 50 columns per table to avoid row size issues
                tables_created = split_wide_table_concatenation(database, table_names, new_table_name, max_columns_per_table)
                
                return jsonify(success=True, 
                              message=f"Successfully split concatenation into {tables_created} tables " +
                                      f"({new_table_name}_1, {new_table_name}_2, etc.) " +
                                      f"to handle the {total_columns} total columns.")
            except Exception as e:
                print(f"Error in split table approach: {e}. Falling back to regular processing.")
                traceback.print_exc()
                # Continue to regular processing as fallback
        
        # If we have a very large number of columns, use the large column process
        if total_columns > MAX_COLUMNS_PER_BATCH and not force_single_table:
            print(f"Very wide table detected ({total_columns} columns). Using optimized approach for large column count.")
            try:
                # MODIFIED: Increase max_columns_per_batch to avoid splitting tables
                max_columns_per_batch = MAX_COLUMNS_PER_BATCH  # Use a very large value to keep all columns together
                
                # Use optimized version for large column count
                process_large_column_concatenation(
                    database, table_names, new_table_name, 
                    {}, all_column_info, row_count, max_columns_per_batch
                )
                
                return jsonify(success=True, 
                              message=f"Successfully concatenated tables into {new_table_name} " +
                                      f"with specialized processing for {total_columns} columns.")
            except Exception as e:
                print(f"Error in large column process: {e}. Falling back to regular processing.")
                traceback.print_exc()
                # Continue to regular processing as fallback
        
        # Regular processing for tables that aren't extremely wide
        print("Using standard concatenation approach.")
        
        # Create an empty DataFrame with a column for each source table column
        # We'll prefix column names with table index to avoid duplicate column names
        columns = []
        column_mappings = {}  # Track original to new column names
        
        for table_info in all_column_info:
            table_idx = table_info['table_index']
            for col_name in table_info['columns']:
                # Create a unique prefixed column name
                prefixed_name = f"t{table_idx}_{col_name}"
                columns.append(prefixed_name)
                
                # Track the mapping from original to prefixed name
                if col_name not in column_mappings:
                    column_mappings[col_name] = []
                column_mappings[col_name].append(prefixed_name)
        
        # Create an empty DataFrame with these columns
        empty_df = pd.DataFrame(columns=columns)
        
        # Add rows to match the row count in the tables
        if row_count > 0:
            # Create rows with placeholder values
            empty_df = pd.DataFrame(index=range(row_count), columns=columns)
            empty_df = empty_df.fillna('')  # Use empty string instead of NaN
        
        # Process tables in batches to avoid memory issues
        for batch_start in range(0, len(table_names), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(table_names))
            batch = table_names[batch_start:batch_end]
            
            print(f"Processing tables {batch_start+1}-{batch_end} of {len(table_names)}...")
            
            # Process each table in the batch
            for table_idx, table_name in enumerate(batch, start=batch_start+1):
                try:
                    print(f"Loading data from {table_name}...")
                    
                    # Get column names first
                    cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
                    columns = [col[0] for col in cursor.fetchall()]
                    
                    # Fetch all data
                    cursor.execute(f"SELECT * FROM `{table_name}`")
                    rows = cursor.fetchall()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=columns)
                    
                    # Map each column to its prefixed name and add to the empty DataFrame
                    for col_name in df.columns:
                        prefixed_name = f"t{table_idx}_{col_name}"
                        empty_df[prefixed_name] = df[col_name].values
                        
                    print(f"Added {len(df.columns)} columns from {table_name}")
                    
                except Exception as e:
                    print(f"Error processing table {table_name}: {e}")
                    connection.close()
                    return jsonify(success=False, message=f'Error processing table {table_name}: {str(e)}')
        
        # First, create the table structure with one empty row to establish the schema
        if row_count > 0:
            # Use a small sample to create the table structure
            sample_df = empty_df.head(1)
            
            # Instead of using to_sql directly, create a table with TEXT columns
            # to avoid row size limit issues
            try:
                # Generate CREATE TABLE SQL with TEXT columns
                create_table_sql = f"CREATE TABLE `{new_table_name}` ("
                column_defs = []
                
                for col in sample_df.columns:
                    # Use TEXT instead of VARCHAR(255) to store data outside the row
                    # This prevents row size limit issues with wide tables
                    column_defs.append(f"`{col}` TEXT")
                    
                create_table_sql += ", ".join(column_defs)
                # Add more aggressive settings for large rows
                create_table_sql += ") ENGINE=InnoDB ROW_FORMAT=DYNAMIC"
                
                # Set MySQL variables to optimize for wide tables
                cursor.execute("SET SESSION innodb_strict_mode=OFF")
                
                # Drop the table if it exists
                cursor.execute(f"DROP TABLE IF EXISTS `{new_table_name}`")
                
                # Create the table with the optimized format
                cursor.execute(create_table_sql)
                connection.commit()
                
                print(f"Created table {new_table_name} with TEXT columns and ROW_FORMAT=DYNAMIC")
                
                # Now insert the data in chunks
                CHUNK_SIZE = 100  # Process 100 rows at a time
                
                for i in range(0, len(empty_df), CHUNK_SIZE):
                    end_idx = min(i + CHUNK_SIZE, len(empty_df))
                    chunk = empty_df.iloc[i:end_idx]
                    
                    # Build SQL placeholders
                    placeholders = ", ".join(["%s"] * len(chunk.columns))
                    column_names = ", ".join([f"`{col}`" for col in chunk.columns])
                    
                    # Build batch insert SQL
                    insert_sql = f"INSERT INTO `{new_table_name}` ({column_names}) VALUES ({placeholders})"
                    
                    # Convert to tuple of tuples for executemany, handling numpy types
                    values = []
                    for row_array in chunk.values: # row_array is a 1D numpy array
                        processed_row = []
                        for item in row_array:
                            if isinstance(item, np.integer):
                                processed_row.append(int(item))
                            elif isinstance(item, np.floating):
                                processed_row.append(float(item))
                            elif pd.isna(item): # Check for pd.NA or np.nan
                                processed_row.append(None) # Convert to SQL NULL
                            else:
                                processed_row.append(str(item)) # Default to string for other types
                        values.append(tuple(processed_row))
                    
                    # Execute the batch insert
                    try:
                        cursor.executemany(insert_sql, values)
                        connection.commit()
                        print(f"Inserted rows {i+1}-{end_idx} of {len(empty_df)}")
                    except Exception as e:
                        print(f"Error inserting rows {i+1}-{end_idx}: {e}")
                        # Try to continue with next chunk
                
                print(f"Successfully inserted all data into {new_table_name}")
                
            except Exception as e:
                print(f"Error creating or populating table: {e}")
                traceback.print_exc()
                cursor.close()
                connection.close()
                return jsonify(success=False, message=f'Error creating or populating table: {str(e)}')
            
        else:
            # For empty tables, just create a table with the structure
            try:
                # Generate CREATE TABLE SQL with TEXT columns
                create_table_sql = f"CREATE TABLE `{new_table_name}` ("
                column_defs = []
                
                for col in empty_df.columns:
                    column_defs.append(f"`{col}` TEXT")
                    
                create_table_sql += ", ".join(column_defs)
                create_table_sql += ") ENGINE=InnoDB ROW_FORMAT=DYNAMIC"
                
                # Drop the table if it exists
                cursor.execute(f"DROP TABLE IF EXISTS `{new_table_name}`")
                
                # Create the empty table
                cursor.execute(create_table_sql)
                connection.commit()
                
                print(f"Created empty table {new_table_name} with {len(empty_df.columns)} columns")
                
            except Exception as e:
                print(f"Error creating empty table: {e}")
                cursor.close()
                connection.close()
                return jsonify(success=False, message=f'Error creating empty table: {str(e)}')
        
        # Close database connection
        cursor.close()
        connection.close()
        
        return jsonify(success=True, message=f"Successfully concatenated {len(table_names)} tables into {new_table_name}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return jsonify(success=False, message=f'Unexpected error: {str(e)}')

def process_large_column_concatenation(database, table_names, new_table_name, 
                                     column_mappings, all_column_info, row_count,
                                     max_columns_per_batch):
    """Process concatenation for tables with a large number of columns by splitting into batches"""
    import pandas as pd
    import numpy as np
    import traceback
    from db_operations import create_long_running_connection, create_long_running_engine
    
    try:
        # Create database connection and engine
        connection = create_long_running_connection(database)
        cursor = connection.cursor()
        engine = create_long_running_engine(database)
        
        # First, create the output table with just the first batch of columns
        # to establish the structure, we'll add more columns later
        
        # Group columns into batches
        all_columns_to_process = []
        for table_info in all_column_info:
            table_name = table_info['table_name']
            table_idx = table_info['table_index']
            for col_name in table_info['columns']:
                all_columns_to_process.append({
                    'table_name': table_name,
                    'table_idx': table_idx,
                    'original_name': col_name,
                    'prefixed_name': f"t{table_idx}_{col_name}"
                })
        
        # MODIFIED: Instead of processing in batches, combine all columns at once
        column_batches = [all_columns_to_process]  # Put all columns in a single batch

        print(f"Processing all {len(all_columns_to_process)} columns together in a single batch")
        
        # Create the output table with all columns
        create_table_sql = f"CREATE TABLE `{new_table_name}` ("
        column_defs = []
        
        # Add all column definitions
        for col_info in all_columns_to_process:
            # Use TEXT instead of VARCHAR(255) for wide tables to avoid row size limits
            # TEXT data types are stored separately with only pointers in the row
            column_defs.append(f"`{col_info['prefixed_name']}` TEXT")
        
        # Complete the CREATE TABLE statement
        create_table_sql += ", ".join(column_defs)
        create_table_sql += ") ENGINE=InnoDB ROW_FORMAT=DYNAMIC"
        
        # Set optimization options
        cursor.execute("SET SESSION innodb_strict_mode=OFF")
        cursor.execute("SET SESSION sql_mode=''")
        # Removed: cursor.execute("SET SESSION innodb_fill_factor=70")
        # Removed: cursor.execute("SET SESSION max_allowed_packet=1073741824")  # Set to 1GB
        # Removed: cursor.execute("SET SESSION innodb_large_prefix=ON")
        # Removed: cursor.execute("SET GLOBAL innodb_file_per_table=ON")
        cursor.execute(f"DROP TABLE IF EXISTS `{new_table_name}`")
        
        # Create the table
        print(f"Creating table {new_table_name} with all {len(column_defs)} columns")
        cursor.execute(create_table_sql)
        connection.commit()
        
        # Now insert the data in chunks
        # Create a DataFrame to hold all rows
        batch_df = pd.DataFrame(index=range(row_count))
        
        # Process each table to fill the DataFrame
        for table_info in all_column_info:
            table_name = table_info['table_name']
            table_idx = table_info['table_index']
            print(f"Loading data from table {table_name}")
            
            # Query the source table for all data
            cursor.execute(f"SELECT * FROM `{table_name}`")
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            # Create a DataFrame from the rows
            source_df = pd.DataFrame(rows, columns=columns)
            
            # Map each column to the target DataFrame with prefixed names
            for col_name in columns:
                prefixed_name = f"t{table_idx}_{col_name}"
                batch_df[prefixed_name] = source_df[col_name]
            
            print(f"Added {len(columns)} columns from {table_name}")
            
            # Clear memory
            del source_df, rows
        
        # Insert the data in chunks to avoid timeouts
        CHUNK_SIZE = 100
        total_rows = len(batch_df)
        
        print(f"Inserting {total_rows} rows in chunks of {CHUNK_SIZE}")
        
        for chunk_start in range(0, total_rows, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_rows)
            chunk = batch_df.iloc[chunk_start:chunk_end]
            
            # Convert the chunk to a list of dictionaries
            records = []
            for _, row in chunk.iterrows():
                row_dict = {}
                for col in row.index:
                    # Handle NaN values
                    val = row[col]
                    if pd.isna(val):
                        row_dict[col] = None
                    else:
                        row_dict[col] = str(val)
                records.append(row_dict)
            
            # Generate placeholders and column names for SQL
            if records:
                col_names = list(records[0].keys())
                placeholders = ", ".join(["%s"] * len(col_names))
                columns_str = ", ".join([f"`{col}`" for col in col_names])
                
                # Create INSERT statement
                insert_sql = f"INSERT INTO `{new_table_name}` ({columns_str}) VALUES ({placeholders})"
                
                # Prepare values for each row
                values = []
                for record in records:
                    row_values = [record[col] for col in col_names]
                    values.append(tuple(row_values))
                
                # Execute the insert
                cursor.executemany(insert_sql, values)
                connection.commit()
                
                print(f"Inserted rows {chunk_start+1} to {chunk_end} of {total_rows}")
        
        print(f"Successfully created and populated table {new_table_name}")
        
        # Close connections
        cursor.close()
        connection.close()
        
        return jsonify(success=True, 
                      message=f"Successfully concatenated tables into {new_table_name}")
        
    except Exception as e:
        print(f"Error in process_large_column_concatenation: {e}")
        traceback.print_exc()
        
        # Close connections
        try:
            if 'cursor' in locals() and cursor is not None:
                cursor.close()
            if 'connection' in locals() and connection is not None:
                connection.close()
        except:
            pass
        
        return jsonify(success=False, 
                      message=f"Error processing large column concatenation: {str(e)}")

def infer_mysql_type_from_pandas(series):
    """Infer an appropriate MySQL data type from a pandas Series"""
    import pandas as pd
    import numpy as np
    
    # Remove None values for type detection
    non_null_series = series.dropna()
    
    # If no non-null values, default to TEXT
    if len(non_null_series) == 0:
        return "TEXT"
    
    # Check the pandas dtype
    dtype = series.dtype
    
    # Numeric types
    if pd.api.types.is_integer_dtype(dtype):
        max_val = series.max() if not pd.isna(series.max()) else 0
        min_val = series.min() if not pd.isna(series.min()) else 0
        
        if min_val >= 0:
            if max_val < 256:
                return "TINYINT UNSIGNED"
            elif max_val < 65536:
                return "SMALLINT UNSIGNED"
            elif max_val < 16777216:
                return "MEDIUMINT UNSIGNED"
            elif max_val < 4294967296:
                return "INT UNSIGNED"
            else:
                return "BIGINT UNSIGNED"
        else:
            if min_val > -128 and max_val < 128:
                return "TINYINT"
            elif min_val > -32768 and max_val < 32768:
                return "SMALLINT"
            elif min_val > -8388608 and max_val < 8388608:
                return "MEDIUMINT"
            elif min_val > -2147483648 and max_val < 2147483648:
                return "INT"
            else:
                return "BIGINT"
    
    elif pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    
    # String types - Always use TEXT for tables with many columns to avoid row size issues
    elif pd.api.types.is_string_dtype(dtype):
        # We'll always use TEXT for string columns to avoid row size issues
        return "TEXT"
    
    # DateTime types
    elif pd.api.types.is_datetime64_dtype(dtype):
        return "DATETIME"
    
    # Default to TEXT for any other type
    return "TEXT"

@app.route('/rename-table', methods=['POST'])
def rename_table():
    data = request.get_json()
    database = data.get('database')
    old_name = data.get('old_name')
    new_name = data.get('new_name')

    if not (database and old_name and new_name):
        return 'Missing required information', 400

    try:
        # Call function to rename table in the database
        result = rename_table_in_database(database, old_name, new_name)
        if result:
            return 'Table renamed successfully', 200
        else:
            return 'Error renaming table', 400
    except Exception as e:
        print(f'Error in rename_table route: {e}')
        return str(e), 500

@app.route('/run-forming-progress')
def run_forming_progress():
    try:
        # Run the script and capture the output
        script_path = '/home/admin2/webapp_2/postprocess/forming_progress.py'
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/home/admin2/webapp_2/postprocess'  # Optional: set working directory
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            output = f"Script exited with return code {process.returncode}\n"
            output += f"Standard Output:\n{stdout}\n"
            output += f"Standard Error:\n{stderr}\n"
        else:
            output = stdout

        # Render the output in a template
        return render_template('forming_progress_output.html', output=output)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"An error occurred while running the script: {e}\n{error_details}", 500

@app.route('/simple_combine', methods=['POST'])
def simple_combine():
    data = request.get_json()
    database = data.get('database')
    table_name = data.get('tableName')
    new_table_name = data.get('newTableName')

    if not (database and table_name and new_table_name):
        return jsonify(success=False, message='Missing required information.')

    try:
        # Connect to the database
        connection = create_connection(database)
        cursor = connection.cursor()

        # Fetch data from the table
        cursor.execute(f"SELECT * FROM `{table_name}`")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

        # Stack all columns into one Series and reset the index
        combined_series = df.stack().reset_index(drop=True)

        # Create a new DataFrame with the combined data
        combined_df = pd.DataFrame({'Combined': combined_series})

        # Check if the new table name already exists
        cursor.execute("SHOW TABLES LIKE %s", (new_table_name,))
        if cursor.fetchone():
            connection.close()
            return jsonify(success=False, message='A table with the new name already exists.')

        # Save the combined DataFrame to a new table using SQLAlchemy engine
        engine = create_db_engine(database)
        combined_df.to_sql(new_table_name, con=engine, if_exists='fail', index=False)

        connection.close()

        return jsonify(success=True)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False, message='An error occurred while combining the table.')

@app.route('/check_single_column_tables', methods=['POST'])
def check_single_column_tables():
    data = request.get_json()
    database = data.get('database')
    table_names = data.get('tableNames')

    if not (database and table_names):
        return jsonify(success=False, message='Missing required information.')

    try:
        connection = create_connection(database)
        cursor = connection.cursor()

        for table_name in table_names:
            cursor.execute(f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = %s AND table_name = %s", (database, table_name))
            column_count = cursor.fetchone()[0]
            if column_count != 1:
                connection.close()
                return jsonify(success=False, message=f"Table '{table_name}' does not have exactly one column.")

        connection.close()
        return jsonify(success=True)
    except Exception as e:
        print(f'Error: {e}')
        return jsonify(success=False, message='An error occurred while checking tables.')

@app.route('/combine_single_columns', methods=['POST'])
def combine_single_columns():
    data = request.get_json()
    database = data.get('database')
    table_names = data.get('tableNames')
    new_table_name = data.get('newTableName')

    if not (database and table_names and new_table_name):
        return jsonify(success=False, message='Missing required information.')

    try:
        # Connect to the database
        connection = create_connection(database)
        cursor = connection.cursor()

        combined_data = pd.DataFrame()

        for table_name in table_names:
            # Fetch the single column from the table
            cursor.execute(f"SELECT * FROM `{table_name}`")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            column_name = table_name  # Use table name as the column header
            combined_data[column_name] = df.iloc[:, 0]  # Assuming the first column

        # Check if the new table name already exists
        cursor.execute("SHOW TABLES LIKE %s", (new_table_name,))
        if cursor.fetchone():
            connection.close()
            return jsonify(success=False, message='A table with the new name already exists.')

        # Save the combined DataFrame to a new table using SQLAlchemy engine
        engine = create_db_engine(database)
        combined_data.to_sql(new_table_name, con=engine, if_exists='fail', index=False)

        connection.close()
        return jsonify(success=True)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(success=False, message='An error occurred while combining the tables.')

@app.route('/check_two_column_table', methods=['POST'])
def check_two_column_table():
    data = request.get_json()
    database = data.get('database')
    table_name = data.get('tableName')

    if not (database and table_name):
        return jsonify(success=False, message='Missing required information.')

    try:
        connection = create_connection(database)
        cursor = connection.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = %s AND table_name = %s
        """, (database, table_name))
        column_count = cursor.fetchone()[0]

        if column_count != 2:
            connection.close()
            return jsonify(success=False, message=f"Table '{table_name}' does not have exactly two columns.")

        connection.close()
        return jsonify(success=True)
    except Exception as e:
        print(f'Error: {e}')
        return jsonify(success=False, message='An error occurred while checking the table.')

@app.route('/generate_scatter_plot/<database>/<table_name>', methods=['GET'])
def generate_scatter_plot(database, table_name):
    try:
        x_column = request.args.get('x_column')
        y_column = request.args.get('y_column')

        if not x_column or not y_column:
            return "X and Y columns must be specified.", 400

        connection = create_connection(database)
        cursor = connection.cursor()

        # Fetch data from the table using the selected columns
        query = f"SELECT `{x_column}`, `{y_column}` FROM `{table_name}`"
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        df = pd.DataFrame(rows, columns=columns)

        # Generate scatter plot
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(df[x_column], df[y_column],
                   label=f'{y_column} vs {x_column}',
                   color='blue',
                   s=1)  # Adjust 's' to make dots smaller

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"Scatter Plot of {table_name}")
        ax.legend()

        # Set the aspect ratio of the plot to be equal
        ax.set_aspect('equal', adjustable='box')

        # Convert plot to PNG image
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        plt.close(fig)
        connection.close()

        return send_file(img, mimetype='image/png')
    except Exception as e:
        print(f'Error: {e}')
        return 'An error occurred while generating the scatter plot.', 500

@app.route('/get_table_columns', methods=['POST'])
def get_table_columns():
    data = request.get_json()
    database = data.get('database')
    table_name = data.get('tableName')

    if not (database and table_name):
        return jsonify(success=False, message='Missing required information.')

    try:
        connection = create_connection(database)
        cursor = connection.cursor()
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        columns = [row[0] for row in cursor.fetchall()]
        connection.close()
        return jsonify(success=True, columns=columns)
    except Exception as e:
        print(f'Error: {e}')
        return jsonify(success=False, message='An error occurred while fetching columns.')

@app.route('/merge-schemas', methods=['POST'])
def merge_schemas():
    try:
        data = request.get_json()
        new_schema_name = data.get('newSchemaName')
        selected_schemas = data.get('selectedSchemas')
        use_existing_schema = data.get('useExistingSchema', False)

        # Validate input parameters
        if not new_schema_name or not selected_schemas:
            return jsonify({'success': False, 'message': 'Missing schema name or source folders'})
        
        # When using a new schema, require at least one source schema
        # When using existing schema, require at least one source schema (can't merge a schema into itself)
        if len(selected_schemas) < 1:
            return jsonify({'success': False, 'message': 'Please select at least one source folder'})

        # MySQL has a table name limit of 64 characters
        MAX_TABLE_NAME_LENGTH = 64
        
        # Create or use existing schema
        connection = create_connection()
        cursor = connection.cursor()
        
        # Check if schema exists
        cursor.execute("SHOW DATABASES LIKE %s", (new_schema_name,))
        schema_exists = cursor.fetchone() is not None
        
        # Handle schema creation based on option
        if not use_existing_schema:
            # For new schemas, verify it doesn't exist
            if schema_exists:
                connection.close()
                return jsonify({'success': False, 'message': 'A folder with this name already exists'})
            
            # Create new schema
            cursor.execute(f"CREATE DATABASE `{new_schema_name}`")
        else:
            # For existing schemas, verify it does exist
            if not schema_exists:
                connection.close()
                return jsonify({'success': False, 'message': 'Target folder does not exist'})
            
            # Make sure we're not trying to merge a schema into itself
            if new_schema_name in selected_schemas:
                connection.close()
                return jsonify({'success': False, 'message': 'Cannot merge a folder into itself'})

        # Keep track of skipped tables
        skipped_tables = []
        processed_tables = []
        duplicate_tables = []
        error_tables = []

        # First, get all existing tables in the target schema to check for conflicts
        target_tables = set()
        if schema_exists:
            cursor.execute(f"SHOW TABLES FROM `{new_schema_name}`")
            target_tables = {table[0] for table in cursor.fetchall()}
        
        # Next, track table names across all selected schemas to identify duplicates
        table_origins = {}  # Dictionary to track which schema each table comes from
        conflicting_tables = set()  # Set to track tables that appear in multiple schemas
        
        for schema in selected_schemas:
            cursor.execute(f"SHOW TABLES FROM `{schema}`")
            tables = cursor.fetchall()
            for (table_name,) in tables:
                if table_name in table_origins:
                    # This table name appears in multiple schemas - mark as conflicting
                    conflicting_tables.add(table_name)
                    table_origins[table_name] = f"{table_origins[table_name]}, {schema}"
                else:
                    table_origins[table_name] = schema

        # Set global MySQL optimization settings for processing wide tables
        try:
            cursor.execute("SET SESSION innodb_strict_mode=OFF")
            cursor.execute("SET SESSION sql_mode=''")
            cursor.execute("SET SESSION max_allowed_packet=1073741824")  # 1GB
            cursor.execute("SET SESSION net_buffer_length=1000000")  # 1MB
            cursor.execute("SET SESSION group_concat_max_len=18446744073709551615")  # Max value
        except Exception as e:
            print(f"Warning: Could not set some MySQL optimization settings: {e}")
            # Continue anyway as these are optimizations, not requirements

        # Process each selected schema
        for schema in selected_schemas:
            # Get all tables from current schema
            cursor.execute(f"SHOW TABLES FROM `{schema}`")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                # Use original table name without schema prefix
                new_table_name = table_name
                
                # Check if the table name exceeds MySQL's limit
                if len(new_table_name) > MAX_TABLE_NAME_LENGTH:
                    skipped_tables.append(f"{schema}.{table_name}")
                    continue
                
                # Skip conflicting tables (same name in multiple source schemas)
                if table_name in conflicting_tables:
                    duplicate_tables.append(f"{schema}.{table_name} (conflicts with tables from: {table_origins[table_name]})")
                    continue
                
                # Check if table already exists in target schema
                if new_table_name in target_tables:
                    duplicate_tables.append(f"{schema}.{table_name} (already exists in target folder)")
                    continue
                
                try:
                    # For very large tables, use optimized copy approach
                    # Get table size information
                    cursor.execute(f"SELECT COUNT(*) FROM `{schema}`.`{table_name}`")
                    row_count = cursor.fetchone()[0]
                    
                    cursor.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = %s AND table_name = %s", 
                                (schema, table_name))
                    column_count = cursor.fetchone()[0]
                    
                    # Add table to the set of tables in target to prevent future conflicts
                    target_tables.add(new_table_name)
                    
                    # Instead of using CREATE TABLE LIKE, create a new table with all columns as TEXT
                    # This will avoid row size limit issues for wide tables
                    cursor.execute(f"""
                        SELECT COLUMN_NAME 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                        ORDER BY ORDINAL_POSITION
                    """, (schema, table_name))
                    
                    columns = [row[0] for row in cursor.fetchall()]
                    
                    # Create the new table with all columns as TEXT type
                    create_table_query = f"CREATE TABLE `{new_schema_name}`.`{new_table_name}` ("
                    column_defs = []
                    
                    for col in columns:
                        # Use TEXT type for all columns to avoid row size issues
                        column_defs.append(f"`{col}` TEXT")
                        
                    create_table_query += ", ".join(column_defs)
                    # Add explicit ROW_FORMAT=DYNAMIC for better handling of wide rows
                    create_table_query += ") ENGINE=InnoDB ROW_FORMAT=DYNAMIC"
                    
                    # Execute the CREATE TABLE statement
                    cursor.execute(create_table_query)
                    
                    # Use batched inserts for all tables, with size based on table dimensions
                    batch_size = 5000  # Default batch size for normal tables
                    if column_count > 100:
                        batch_size = 1000  # Medium size tables
                    if column_count > 500:
                        batch_size = 500  # Large tables
                    if column_count > 1000:
                        batch_size = 100  # Extra large tables
                        
                    # Build column list string for SELECT and INSERT
                    columns_str = ", ".join([f"`{col}`" for col in columns])
                    
                    # Use batched inserts
                    for offset in range(0, row_count, batch_size):
                        limit = min(batch_size, row_count - offset)
                        insert_query = f"""
                            INSERT INTO `{new_schema_name}`.`{new_table_name}` ({columns_str})
                            SELECT {columns_str} FROM `{schema}`.`{table_name}` LIMIT {limit} OFFSET {offset}
                        """
                        cursor.execute(insert_query)
                        connection.commit()
                    
                    processed_tables.append(f"{schema}.{table_name}")
                    
                except mysql.connector.Error as err:
                    # Track specific MySQL errors
                    error_tables.append(f"{schema}.{table_name} (MySQL Error: {err})")
                    print(f"Error copying table {schema}.{table_name}: {err}")
                    # Remove from target_tables since it failed
                    if new_table_name in target_tables:
                        target_tables.remove(new_table_name)
                    # Try to clean up the partially created table if it exists
                    try:
                        cursor.execute(f"DROP TABLE IF EXISTS `{new_schema_name}`.`{new_table_name}`")
                        connection.commit()
                    except:
                        pass
                except Exception as e:
                    # Track general errors
                    error_tables.append(f"{schema}.{table_name} (Error: {str(e)})")
                    print(f"Unexpected error copying table {schema}.{table_name}: {e}")
                    # Remove from target_tables since it failed
                    if new_table_name in target_tables:
                        target_tables.remove(new_table_name)
                    # Try to clean up the partially created table if it exists
                    try:
                        cursor.execute(f"DROP TABLE IF EXISTS `{new_schema_name}`.`{new_table_name}`")
                        connection.commit()
                    except:
                        pass

        connection.commit()
        cursor.close()
        connection.close()
        
        # Prepare response message
        result = {'success': True}
        message_parts = []
        
        if processed_tables:
            message_parts.append(f'Successfully merged {len(processed_tables)} tables.')
        else:
            message_parts.append('No tables were merged.')
            
        if skipped_tables:
            message_parts.append(f'{len(skipped_tables)} tables were skipped due to name length exceeding the 64-character limit.')
            result['skipped_tables'] = skipped_tables
            
        if duplicate_tables:
            message_parts.append(f'{len(duplicate_tables)} tables were skipped due to name conflicts between source folders or existing in the target folder.')
            result['duplicate_tables'] = duplicate_tables
            
        if error_tables:
            message_parts.append(f'{len(error_tables)} tables encountered errors during the merge process.')
            result['error_tables'] = error_tables
            
        result['message'] = ' '.join(message_parts)
        return jsonify(result)
        
    except Exception as e:
        print(f'Error merging schemas: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'An error occurred: {str(e)}'})

#------------------------------------------------------------------------------------------------------------

def sanitize_table_name(name):
    """
    Sanitize the filename to make it suitable for usage as a MySQL table name.
    """
    # Remove non-word characters and spaces
    sanitized_name = re.sub(r'\W+| ', '_', name)

    # Ensure it starts with a letter, prepend an 'a' if not
    #if not sanitized_name[0].isalpha():
        #sanitized_name = 'a' + sanitized_name

    return sanitized_name.lower()

def validate_filename(filename):
    print("Filename:", filename)
    #pattern = r"^lot[A-Za-z0-9]+_wafer[A-Za-z0-9]+_die[A-Za-z0-9]+_dut[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+\.(csv|txt)$"
    pattern = r"^lot[A-Za-z0-9]+_wafer[A-Za-z0-9]+_die[A-Za-z0-9]+_dut[A-Za-z0-9]*_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+_[A-Za-z0-9]+\.(csv|txt|npy)$"
    print("Pattern match result:", bool(re.match(pattern, filename)))
    return bool(re.match(pattern, filename))

def render_results(results):
    results_html = "<html><body style='background-color: white;'>"
    for result in results:
        results_html += f"<p>{result}</p>"
    results_html += "</body></html>"
    return render_template_string(results_html), 200

def get_form_data_generate_plot(form):
    form_data = {
        key: form.get(key, "").strip() for key in [
            'state_pattern_type', 'number_of_states',
            'selected_groups_1D', 'pass_range_1D', 'state_pattern',
            'selected_groups_predefined', 'pass_range_predefined',
            'custom_selected_groups_predefined', 'custom_pass_range_predefined',
            'color_map_flag', 'outlier_analysis_flag', 'target_values', 'custom_division_type', 'color_group_keywords',
            'target_x_diff', 'num_interp_points', 'ber_lower_limit', 'ber_upper_limit', 'top_ios_count', 'ber_display_option',  # Added ber_display_option field
            'data_min_value', 'data_max_value',  # Added data range filter fields
            'filter_negative_values',  # Added negative value filter field
            'generate_bitmap_mask', 'bitmap_mask_name', 'apply_bitmap_mask',  # Added bitmap mask fields
            'analysis_type',  # Added analysis type field for column-by-column analysis
            'column_selection_type', 'custom_column_selection', 'yanCullinan_flag',  # Added column selection fields
            'location_dots_flag', 'location_dots_value'  # Added location dots fields
        ]
    }

    print("Form Data Retrieved:", form_data)  # Debug print

    if form_data['state_pattern_type'] == '1D':
        form_data['number_of_states'] = int(form_data['number_of_states']) if form_data['number_of_states'].isdigit() else None
        # Handle selected_groups
        if form_data.get('selected_groups_1D') == 'custom' and form.get('custom_selected_groups_1D'):
            selected_groups_1D = form.get('custom_selected_groups_1D').split(',')
        else:
            selected_groups_1D = form_data.get('selected_groups_1D', '').split(',') if form_data.get('selected_groups_1D') else []
        form_data['selected_groups'] = [int(float(num)) for num in selected_groups_1D if num.strip()]
    elif form_data['state_pattern_type'] == 'predefined':
        # Handle selected_groups
        if form_data.get('selected_groups_predefined') == 'custom' and form_data.get('custom_selected_groups_predefined'):
            selected_groups_predefined = form_data.get('custom_selected_groups_predefined', '').split(',')
        else:
            selected_groups_predefined = form_data.get('selected_groups_predefined', '').split(',') if form_data.get('selected_groups_predefined') else []
        
        form_data['selected_groups'] = [int(float(num)) for num in selected_groups_predefined if num.strip()]
        print('selected_groups:',  form_data['selected_groups'])

    form_data['state_pattern'] = form.get('state_pattern', None)
    
    # Handle pass_range
    if form_data.get('pass_range_predefined') == 'custom' and form_data.get('custom_pass_range_predefined'):
        pass_range_predefined = form_data.get('custom_pass_range_predefined', '').split(',')
    else:
        pass_range_predefined = form_data.get('pass_range_predefined', '').split(',') if form_data.get('pass_range_predefined') else []
    
    form_data['pass_range'] = [float(num) for num in pass_range_predefined if num.strip()]

    # Convert checkbox flags to boolean
    form_data['color_map_flag'] = form_data.get('color_map_flag', 'False') == 'True'
    form_data['yanCullinan_flag'] = form_data.get('yanCullinan_flag', 'False') == 'True'
    form_data['outlier_analysis_flag'] = form_data.get('outlier_analysis_flag', 'False') == 'True'
    form_data['filter_negative_values'] = form_data.get('filter_negative_values', 'False') == 'True'
    form_data['generate_bitmap_mask'] = form_data.get('generate_bitmap_mask', 'False') == 'True'
    form_data['location_dots_flag'] = form_data.get('location_dots_flag', 'False') == 'True'
    
    # Process location dots value
    location_dots_value_str = form_data.get('location_dots_value', '')
    if location_dots_value_str:
        try:
            form_data['location_dots_value'] = int(location_dots_value_str)
        except ValueError:
            form_data['location_dots_value'] = None
    else:
        form_data['location_dots_value'] = None
    
    # Handle custom division - convert from dropdown selection to boolean and values
    custom_division_type = form_data.get('custom_division_type', '')
    if custom_division_type:
        form_data['custom_division'] = True
        # Parse the custom division values from the selected option
        try:
            form_data['custom_division_values'] = [int(x.strip()) for x in custom_division_type.split(',')]
        except ValueError:
            form_data['custom_division'] = False
            form_data['custom_division_values'] = []
    else:
        form_data['custom_division'] = False
        form_data['custom_division_values'] = []

    # Process target values
    target_values_str = form_data.get('target_values', '')
    if target_values_str:
        try:
            form_data['target_values'] = [float(x.strip()) for x in target_values_str.split(',') if x.strip()]
        except ValueError:
            form_data['target_values'] = []
    else:
        form_data['target_values'] = []

    # Process color grouping keywords
    color_group_keywords_str = form_data.get('color_group_keywords', '')
    if color_group_keywords_str:
        form_data['color_group_keywords'] = [keyword.strip() for keyword in color_group_keywords_str.split(',') if keyword.strip()]
    else:
        form_data['color_group_keywords'] = []
    
    # Process target_x_diff value
    target_x_diff_str = form.get('target_x_diff', '2')
    print(f"Raw target_x_diff from form: '{target_x_diff_str}'")
    try:
        form_data['target_x_diff'] = float(target_x_diff_str) if target_x_diff_str else 2.0
        print(f"Converted target_x_diff to: {form_data['target_x_diff']}")
    except ValueError:
        form_data['target_x_diff'] = 2.0  # Default to 2.0 if conversion fails
        print(f"Failed to convert target_x_diff, using default: {form_data['target_x_diff']}")
    
    # Process num_interp_points value
    num_interp_points_str = form.get('num_interp_points', '1000')
    print(f"Raw num_interp_points from form: '{num_interp_points_str}'")
    try:
        form_data['num_interp_points'] = int(num_interp_points_str) if num_interp_points_str else 500
        # Remove constraints so users can use any value
        print(f"Converted num_interp_points to: {form_data['num_interp_points']}")
    except ValueError:
        form_data['num_interp_points'] = 500  # Default to 500 if conversion fails
        print(f"Failed to convert num_interp_points, using default: {form_data['num_interp_points']}")

    # Process top_ios_count value
    top_ios_count_str = form.get('top_ios_count', '')
    print(f"Raw top_ios_count from form: '{top_ios_count_str}'")
    try:
        if top_ios_count_str.strip():
            top_ios_count = int(top_ios_count_str)
            # Ensure it's at least 1 if provided
            form_data['top_ios_count'] = max(1, top_ios_count)
            print(f"Converted top_ios_count to: {form_data['top_ios_count']}")
        else:
            # If empty, set to None to display all tables
            form_data['top_ios_count'] = None
            print("No top_ios_count provided, will display all tables")
    except ValueError:
        # Default to None if conversion fails
        form_data['top_ios_count'] = None
        print("Failed to convert top_ios_count, will display all tables")

    # Process BER range limits
    ber_lower_limit_str = form.get('ber_lower_limit', '')
    ber_upper_limit_str = form.get('ber_upper_limit', '')
    
    try:
        form_data['ber_lower_limit'] = int(ber_lower_limit_str) if ber_lower_limit_str.strip() else None
        print(f"Converted ber_lower_limit to: {form_data['ber_lower_limit']}")
    except ValueError:
        form_data['ber_lower_limit'] = None
        print(f"Failed to convert ber_lower_limit, using None")
    
    try:
        form_data['ber_upper_limit'] = int(ber_upper_limit_str) if ber_upper_limit_str.strip() else None
        print(f"Converted ber_upper_limit to: {form_data['ber_upper_limit']}")
    except ValueError:
        form_data['ber_upper_limit'] = None
        print(f"Failed to convert ber_upper_limit, using None")

    # Set default for ber_display_option if not provided
    if 'ber_display_option' not in form_data or not form_data['ber_display_option']:
        form_data['ber_display_option'] = 'top_ios'  # Default to top_ios if not specified

    # Process data range filter values
    data_min_value_str = form.get('data_min_value', '')
    data_max_value_str = form.get('data_max_value', '')
    
    try:
        form_data['data_min_value'] = float(data_min_value_str) if data_min_value_str.strip() else None
        print(f"Converted data_min_value to: {form_data['data_min_value']}")
    except ValueError:
        form_data['data_min_value'] = None
        print(f"Failed to convert data_min_value, using None")
    
    try:
        form_data['data_max_value'] = float(data_max_value_str) if data_max_value_str.strip() else None
        print(f"Converted data_max_value to: {form_data['data_max_value']}")
    except ValueError:
        form_data['data_max_value'] = None
        print(f"Failed to convert data_max_value, using None")

    # Process column selection for column-by-column analysis
    if (form_data.get('state_pattern') == '82944x78_ecc_fuxi' and 
        form_data.get('analysis_type') == 'column_by_column'):
        
        column_selection_type = form_data.get('column_selection_type', 'all')
        if column_selection_type == 'custom':
            custom_selection = form_data.get('custom_column_selection', '').strip()
            if custom_selection:
                try:
                    form_data['selected_columns'] = parse_column_selection(custom_selection)
                except ValueError as e:
                    print(f"Error parsing column selection: {e}")
                    form_data['selected_columns'] = list(range(78))  # Default to all columns
            else:
                form_data['selected_columns'] = list(range(78))  # Default to all columns
        else:
            form_data['selected_columns'] = list(range(78))  # All columns (0-77)
    else:
        form_data['selected_columns'] = list(range(78))  # Default for non-column analysis

    print("Final Form Data:", form_data)  # Debug print
    return form_data

def parse_column_selection(selection_str):
    """
    Parse column selection string into list of column indices.
    Supports formats like: "0,1,2", "0-5", "1-10,45,23", etc.
    """
    selected_columns = []
    parts = selection_str.replace(' ', '').split(',')
    
    for part in parts:
        if '-' in part:
            # Handle range (e.g., "0-5")
            range_parts = part.split('-')
            if len(range_parts) == 2:
                start = int(range_parts[0])
                end = int(range_parts[1])
                if 0 <= start <= 77 and 0 <= end <= 77 and start <= end:
                    selected_columns.extend(range(start, end + 1))
                else:
                    raise ValueError(f"Invalid range: {part}")
            else:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Handle single number (e.g., "5")
            num = int(part)
            if 0 <= num <= 77:
                selected_columns.append(num)
            else:
                raise ValueError(f"Invalid column number: {num}")
    
    # Remove duplicates and sort
    return sorted(list(set(selected_columns)))

def flatten_sections(array_3d):
    """Flatten 16x16 sections from a 3D array."""
    slices = [array_3d[row:row+16, col:col+16, setting].flatten()
              for setting in range(array_3d.shape[2])
              for row in range(0, 64, 16)
              for col in range(0, 64, 16)]
    return np.concatenate(slices)

def process_mat_file(file_content):
    print('process_mat_file')
    df = None
    mat_data = scipy.io.loadmat(BytesIO(file_content))
    for key in mat_data:
        if not key.startswith('__'):
            data = mat_data[key]
            print("data.shape", data.shape)
            if data.shape == (64, 64, 8) or data.shape == (64, 64, 4) or data.shape == (100, 64, 64):
                df = pd.DataFrame(flattened_data)
            elif data.shape == (1, 64, 64):
                flattened_data = data.squeeze()  # Flattening to (64, 64)
                df = pd.DataFrame(flattened_data)
            elif len(data.shape) == 2:  # Check if the data is 2D
                df = pd.DataFrame(data)
            if df is not None:
                break
    return df

def process_h5py_file(file_stream):  #64x64
    print('process_h5py_file')
    df = None
    with h5py.File(file_stream, 'r') as f:
        for key in f.keys():
            data = f[key]
            if isinstance(data, h5py.Dataset) and data.shape == (64, 64):
                df = pd.DataFrame(data[:]).T
                break
    return df

def is_hdf5_file(file_stream):
    # Check if the file stream is an HDF5 file
    try:
        h5py.File(file_stream, 'r')
        return True
    except OSError:
        return False

import pandas as pd
from pyexcel_xls import get_data
import sys
import json

def process_file(file_stream, file_extension, db_name):
    #print("Python version:", sys.version)
    #print("pandas version:", pd.__version__)
    
    file_stream.seek(0)
    df = None

    if file_extension == "csv":
        try:
            df = pd.read_csv(file_stream, header=None, skiprows=0)
            print("shape of df:", df.shape)
        except Exception as e:
            print("An error occurred:", e)

    elif file_extension == "xlsx":
        print(f"file_extension == {file_extension}") 
        try:
            df = pd.read_excel(file_stream, header=None, skiprows=0, engine='openpyxl')
            print("shape of df:", df.shape)
        except Exception as e:
            print("An error occurred:", e)

    elif file_extension == "xls":
        print(f"file_extension == {file_extension}") 
        try:
            # Read the .xls file using pyexcel
            data = get_data(file_stream)
            # Assuming you want to read the first sheet
            sheet_name = list(data.keys())[0]
            sheet_data = data[sheet_name]
            df = pd.DataFrame(sheet_data)
            print("shape of df:", df.shape)
        except Exception as e:
            print("An error occurred:", e)

    elif file_extension == "txt":
        print("uploading txt")
        content = file_stream.read()
        print(f"Content length: {len(content)}")  # Log the length of the content

        if len(content) > 65535:
            raise ValueError("Content too large to fit in the database column")

        df = pd.DataFrame({'content': [content]})
        print("DataFrame created")  # Log DataFrame creation
        print(df)  # Print the DataFrame

    elif file_extension == "npy":
        try:
            # Load the .npy file into a numpy array
            np_array = np.load(file_stream, allow_pickle=True)
            print("Original array shape:", np_array.shape)

            if np_array.ndim == 1:
                df = pd.DataFrame(np_array)
            elif np_array.ndim == 2:
                df = pd.DataFrame(data=np_array)
                if df.shape[1] > 1017:  #df.shape[1] is the num of columns
                    df = df.transpose()  #to transpose 64x1296 to 1296x64              
            elif np_array.ndim == 3:
                df = pd.DataFrame(data=np_array.reshape(np_array.shape[0], -1))
            elif np_array.ndim == 4:
                squeezed_array = np.squeeze(np_array)
                if squeezed_array.ndim < 4:
                    np_array = squeezed_array
                    print("Array was squeezed to dimensions:", np_array.shape)
                else:
                    print("Squeezing did not reduce dimensions, handling as 4D array.")
                df = pd.DataFrame(data=np_array.reshape(-1, np_array.shape[-1]))
            else:
                raise ValueError("Numpy array dimensionality not supported")

            # Check for NaN values immediately after DataFrame creation
            if df.isnull().values.any():
                print("DataFrame contains NaN values after creation.")
            else:
                print("DataFrame does not contain NaN values after creation.")

            print("DataFrame dtypes:")
            print(df.dtypes)

        except Exception as e:
            print(f"Error processing .npy file: {e}")
            df = pd.DataFrame()

    elif file_extension == "mat":
        # Reset the file stream to the beginning for reading
        file_stream.seek(0)
        if is_hdf5_file(file_stream):
            # If it's an HDF5 file, use the h5py processor
            df = process_h5py_file(file_stream)
        else:
            # For other .mat files, process them here
            try:
                # Reset the file stream again as is_hdf5_file may have moved it
                file_stream.seek(0)
                file_content = file_stream.read()
                df = process_mat_file(file_content)
                if df is None or df.empty:
                    raise ValueError("No suitable dataset found in the .mat file")
            except Exception as e:
                print(f"Error processing .mat file: {e}")
                df = pd.DataFrame()
    
    elif file_extension == "json":
        print(f"file_extension == {file_extension}")
        try:
            file_stream.seek(0)
            df = pd.read_json(file_stream)
            print("shape of df:", df.shape)
        except ValueError as e:
            print("An error occurred with pd.read_json:", e)
            try:
                file_stream.seek(0)
                content = file_stream.read()
                # If content is bytes, decode to string
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                # Load JSON data
                data = json.loads(content)
                # Normalize JSON data to a flat table
                df = pd.json_normalize(data)
                print("shape of df after normalization:", df.shape)
            except Exception as e:
                print("An error occurred while processing JSON data:", e)
                df = pd.DataFrame()
        except Exception as e:
            print("An unexpected error occurred:", e)
            df = pd.DataFrame()

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    # Fallback for any unprocessed or empty data frames
    if df is None or df.empty:
        print("No data processed for the file, returning empty DataFrame")
        df = pd.DataFrame()
    
    return df

# Add custom Jinja2 filter for NaN values
@app.template_filter('is_nan')
def is_nan_filter(value):
    try:
        return np.isnan(value)
    except:
        return False

def get_disk_space():
    """Get the disk space information for the MySQL data directory."""
    try:
        # Get MySQL data directory path (you may need to adjust this path)
        mysql_path = "/"  # Check disk usage on /app mount point to avoid permission issues
        
        # Get disk usage statistics
        disk_stats = shutil.disk_usage(mysql_path)
        
        # Convert to human readable format
        def format_size(size):
            units = ['B', 'KB', 'MB', 'GB', 'TB']
            size = float(size)
            unit_index = 0
            while size >= 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1
            return f"{size:.2f} {units[unit_index]}"
        
        # Format free and total space
        free_space = format_size(disk_stats.free)
        total_space = format_size(disk_stats.total)
        used_space = format_size(disk_stats.used)
        usage_percent = f"{(disk_stats.used / disk_stats.total) * 100:.1f}%"
        
        # Return a dictionary with all disk information
        return {
            "free_space": free_space,
            "total_space": total_space,
            "used_space": used_space,
            "usage_percent": usage_percent
        }
    except Exception as e:
        print(f"Error getting disk space: {e}")
        return {"free_space": "Unknown", "total_space": "Unknown", "used_space": "Unknown", "usage_percent": "Unknown"}

# Jupyter Notebook integration
@app.route('/notebook')
def jupyter_notebook():
    """
    Redirect to the notebook selector page instead of directly showing a specific notebook.
    This will be the entry point for users to access all available notebooks.
    """
    return redirect(url_for('notebook_selector'))

@app.route('/notebook-selector')
def notebook_selector():
    """
    Display a page with a list of available notebooks and an option to create a new one.
    """
    postprocess_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postprocess')
    
    # Check if the directory exists
    if not os.path.exists(postprocess_dir):
        os.makedirs(postprocess_dir)
    
    # Get all .ipynb files in the directory
    notebooks = [f for f in os.listdir(postprocess_dir) if f.endswith('.ipynb')]
    notebooks.sort()  # Sort alphabetically
    
    # Get the modification dates for each notebook
    notebook_dates = []
    for notebook in notebooks:
        file_path = os.path.join(postprocess_dir, notebook)
        mod_time = os.path.getmtime(file_path)
        mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        notebook_dates.append(mod_time_str)
    
    return render_template('notebook_selector.html', notebooks=notebooks, notebook_dates=notebook_dates)

@app.route('/open-notebook/<notebook_name>')
def open_notebook(notebook_name):
    """
    Open a specific Jupyter notebook.
    """
    # Validate the notebook name to prevent directory traversal
    if '..' in notebook_name or '/' in notebook_name or '\\' in notebook_name:
        flash('Invalid notebook name', 'danger')
        return redirect(url_for('notebook_selector'))
    
    # Ensure the notebook exists
    postprocess_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postprocess')
    notebook_path = os.path.join(postprocess_dir, notebook_name)
    
    if not os.path.exists(notebook_path) or not notebook_name.endswith('.ipynb'):
        flash('Notebook not found', 'danger')
        return redirect(url_for('notebook_selector'))
    
    # Check if Jupyter server is running
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.jupyter_status.json')
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            # Check if the process is still running
            pid = status.get('pid')
            if pid:
                try:
                    # On Unix, this will raise an error if the process doesn't exist
                    os.kill(pid, 0)
                    
                    # Process exists, return the notebook page
                    token = status.get('token', '')
                    # Calculate the relative path from notebook_dir to the target notebook
                    notebook_rel_path = os.path.relpath(notebook_path, postprocess_dir)
                    return render_template('jupyter_notebook.html', 
                                           token=token, 
                                           notebook_path=notebook_rel_path,
                                           notebook_name=notebook_name)
                except:
                    # Process doesn't exist anymore
                    pass
        except:
            # Error reading status file
            pass
    
    # If we get here, we need to tell the user to start the Jupyter server first
    return render_template('jupyter_start_instructions.html')

@app.route('/create-notebook', methods=['POST'])
def create_notebook():
    """
    Create a new Jupyter notebook with the given name.
    """
    notebook_name = request.form.get('notebook_name', '')
    
    # Validate the notebook name
    if not notebook_name or not re.match(r'^[a-zA-Z0-9_-]+$', notebook_name):
        flash('Invalid notebook name. Use only letters, numbers, underscores, and hyphens.', 'danger')
        return redirect(url_for('notebook_selector'))
    
    # Add .ipynb extension if not present
    if not notebook_name.endswith('.ipynb'):
        notebook_name = f"{notebook_name}.ipynb"
    
    postprocess_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postprocess')
    notebook_path = os.path.join(postprocess_dir, notebook_name)
    
    # Check if file already exists
    if os.path.exists(notebook_path):
        flash(f'A notebook with the name {notebook_name} already exists.', 'warning')
        return redirect(url_for('open_notebook', notebook_name=notebook_name))
    
    # Create a new notebook file with basic structure
    notebook_content = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Add a default cell with import statements
    notebook_content["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# New Jupyter notebook\n",
            "# Import common libraries\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import os, sys\n",
            "\n",
            "# Add core processing functions\n",
            "import core_post_processing_functions as cf\n",
            "\n",
            "# Your code here"
        ],
        "outputs": []
    })
    
    # Write the notebook to file
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    # Redirect to open the new notebook
    return redirect(url_for('open_notebook', notebook_name=notebook_name))

@app.route('/jupyter/<path:path>')
def jupyter_proxy(path=''):
    """
    Proxy requests to the Jupyter server.
    """
    # Read the token from the file
    token_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.jupyter_token')
    token = ''
    
    try:
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                token = f.read().strip()
    except:
        pass
    
    # Get the server's hostname instead of hardcoded localhost
    hostname = socket.gethostname()
    
    # Build the target URL
    jupyter_url = f'http://{hostname}:8888/{path}'
    
    # Add token if we have one and it's not already in the URL
    if token and 'token=' not in request.query_string.decode('utf-8'):
        jupyter_url += f'?token={token}'
    
    return redirect(jupyter_url)

@app.route('/check-jupyter')
def check_jupyter():
    """API endpoint to check if Jupyter server is running"""
    status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.jupyter_status.json')
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            # Check if the process is still running
            pid = status.get('pid')
            if pid:
                try:
                    # On Unix, this will raise an error if the process doesn't exist
                    os.kill(pid, 0)
                    
                    # Process exists, return the notebook page
                    token = status.get('token', '')
                    # Calculate the relative path from notebook_dir to the target notebook
                    notebook_rel_path = os.path.relpath(notebook_path, postprocess_dir)
                    return render_template('jupyter_notebook.html', 
                                           token=token, 
                                           notebook_path=notebook_rel_path,
                                           notebook_name=notebook_name)
                except:
                    # Process doesn't exist anymore
                    pass
        except:
            # Error reading status file
            pass
    
    # If we get here, we need to tell the user to start the Jupyter server first
    return render_template('jupyter_start_instructions.html')

@app.route('/test-machines')
def test_machines():
    """
    Page to display the available test machines
    """
    test_machines = [
        {'ip': '192.168.68.124', 'user': 'slate', 'hostname': 'slate'},
        {'ip': '192.168.68.234', 'user': 'tc4', 'hostname': 'TC4'},
        {'ip': '192.168.68.129', 'user': 'nuc14', 'hostname': 'NUC14'},
        {'ip': '192.168.68.206', 'user': 'nuc6', 'hostname': 'NUC6'},
        {'ip': '192.168.68.164', 'user': 'lenovoi7', 'hostname': 'lenovoi7'},
        {'ip': '192.168.68.205', 'user': 'nuc5', 'hostname': 'NUC5'},
        {'ip': '192.168.68.235', 'user': 'tc5', 'hostname': 'TC5'},
        {'ip': '192.168.68.231', 'user': 'tc1', 'hostname': 'TC1'}, 
        {'ip': '192.168.68.232', 'user': 'tc2', 'hostname': 'TC2'}
    ]
    return render_template('test_machines.html', test_machines=test_machines)

@app.route('/test-ssh-connection', methods=['POST'])
def test_ssh_connection():
    """
    Test SSH connection to a remote machine
    """
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            
            if not machine_ip or not machine_user:
                return jsonify({
                    "success": False,
                    "message": "Missing required parameters"
                })
            
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
            
            # Try to connect via SSH
            import paramiko
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=machine_ip,
                username=machine_user,
                password=password,
                timeout=10
            )
            
            # Run a simple command to verify connection
            stdin, stdout, stderr = ssh_client.exec_command('hostname')
            hostname = stdout.read().decode('utf-8').strip()
            
            # Close the connection
            ssh_client.close()
            
            return jsonify({
                "success": True,
                "message": f"Successfully connected to {machine_user}@{machine_ip}",
                "output": f"Machine hostname: {hostname}"
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Failed to connect to {machine_user}@{machine_ip}",
                "error": str(e)
            })

@app.route('/execute-ssh-command', methods=['POST'])
def execute_ssh_command():
    """
    Execute a command on a remote machine via SSH and return the result as JSON.
    This is used for the interactive terminal sessions.
    """
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            command = request.form.get('command')
            terminal_id = request.form.get('terminal_id', '1')  # Default to terminal 1
            current_dir = request.form.get('current_directory', '~')  # Get current directory
            
            if not machine_ip or not machine_user or not command:
                return jsonify({
                    "success": False,
                    "message": "Missing required parameters"
                })
            
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
            
            # Check if command is a cd command, if so update the current directory
            new_directory = None
            if command.strip().startswith('cd '):
                path = command.strip()[3:].strip()
                # Handle relative paths
                if not path.startswith('/') and not path.startswith('~'):
                    if current_dir == '~':
                        new_directory = f"~/{path}"
                    else:
                        new_directory = f"{current_dir}/{path}"
                else:
                    new_directory = path
            
            # Build the actual command to execute with the current directory
            if new_directory:
                # If cd command, change directory and show new location
                actual_command = f"cd {new_directory} && pwd && echo ''"
            else:
                # For regular commands, execute them in the current directory
                actual_command = f"cd {current_dir} && {command} && pwd"
            
            # For machines with password, use sshpass to run the command
            if machine_ip in machine_passwords:
                password = machine_passwords[machine_ip]
                result = subprocess.run(
                    ['sshpass', '-p', password, 'ssh', 
                     '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'{machine_user}@{machine_ip}', actual_command],
                    capture_output=True,
                    text=True,
                    timeout=30  # Longer timeout for commands
                )
                
                # Extract the new working directory from the output (last line)
                output_lines = result.stdout.strip().split('\n')
                
                if output_lines:
                    # The last line is the current directory
                    current_dir = output_lines[-1]
                    # Remove the current directory from the output
                    output = '\n'.join(output_lines[:-1])
                else:
                    output = ''
                
                # Special handling for 'cd' commands
                if new_directory:
                    output = f"Changed directory to: {current_dir}"
                
                return jsonify({
                    "success": True,
                    "output": output,
                    "error": result.stderr if result.stderr else None,
                    "current_directory": current_dir  # Return the current directory
                })
            else:
                # For key-based authentication, try without password
                result = subprocess.run(
                    ['ssh', '-o', 'BatchMode=yes', 
                     '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'{machine_user}@{machine_ip}', actual_command],
                    capture_output=True,
                    text=True,
                    timeout=30  # Longer timeout for commands
                )
                
                # Extract the new working directory from the output (last line)
                output_lines = result.stdout.strip().split('\n')
                
                if output_lines:
                    # The last line is the current directory
                    current_dir = output_lines[-1]
                    # Remove the current directory from the output
                    output = '\n'.join(output_lines[:-1])
                else:
                    output = ''
                
                # Special handling for 'cd' commands
                if new_directory:
                    output = f"Changed directory to: {current_dir}"
                
                return jsonify({
                    "success": True,
                    "output": output,
                    "error": result.stderr if result.stderr else None,
                    "current_directory": current_dir  # Return the current directory
                })
                
        except subprocess.TimeoutExpired:
            return jsonify({
                "success": False,
                "message": "Command execution timed out"
            })
        except Exception as e:
            # Handle general errors
            return jsonify({
                "success": False,
                "message": f"Server Error: {str(e)}",
                "error": str(e)
            })

@app.route('/remote-commands')
def remote_commands():
    """
    Display the remote commands interface for a specific machine.
    """
    machine_ip = request.args.get('ip')
    machine_user = request.args.get('user')
    
    if not machine_ip or not machine_user:
        flash('Missing machine information', 'danger')
        return redirect(url_for('test_machines'))
        
    return render_template('remote_commands.html', 
                          machine_ip=machine_ip, 
                          machine_user=machine_user)

@app.route('/execute-remote-command', methods=['POST'])
def execute_remote_command():
    """
    Execute a command on a remote machine via SSH
    """
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            directory_1 = request.form.get('directory_1')
            directory_2 = request.form.get('directory_2')
            command_type = request.form.get('command_type')
            
            if not machine_ip or not machine_user or not (directory_1 or directory_2) or not command_type:
                return jsonify({
                    'success': False, 
                    'message': 'Missing required parameters'
                })
            
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
            
            # Determine which command to run based on the command type
            if command_type == 'serial_run':
                directory = directory_1
                command = 'cd ' + directory + ' && python3 serial_run.py'
            elif command_type == 'test_automation':
                directory = directory_2
                command = 'cd ' + directory + ' && source ~/testenv/bin/activate && bash select_run_test.sh'
            else:
                return jsonify({
                    'success': False, 
                    'message': 'Invalid command type'
                })
            
            # Execute the command on the remote machine
            if machine_ip in machine_passwords:
                password = machine_passwords[machine_ip]
                result = subprocess.run(
                    ['sshpass', '-p', password, 'ssh', 
                     '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'{machine_user}@{machine_ip}', command],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                # Try without password
                result = subprocess.run(
                    ['ssh', '-o', 'BatchMode=yes', 
                     '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'{machine_user}@{machine_ip}', command],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            
            # Return the result
            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'message': 'Command executed successfully',
                    'output': result.stdout.strip()
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Command execution failed',
                    'error': result.stderr.strip()
                })
                
                
        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False, 
                'message': 'Command execution timed out'
            })
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Error: {str(e)}'
            })
    
    return jsonify({'success': False, 'message': 'Invalid request method'})

@app.route('/browse-remote-directory', methods=['POST'])
def browse_remote_directory():
    """
    Fetch directory contents from a remote machine for the file browser.
    """
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            current_path = request.form.get('current_path', '~')
            
            if not machine_ip or not machine_user:
                return jsonify({'success': False, 'message': 'Missing IP or username'})
            
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
            
            # Build the command to list directories and their content
            command = f"find {current_path} -maxdepth 1 -type d | sort"
            
            # Execute the command on the remote machine
            if machine_ip in machine_passwords:
                password = machine_passwords[machine_ip]
                result = subprocess.run(
                    ['sshpass', '-p', password, 'ssh', 
                     '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'{machine_user}@{machine_ip}', command],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
            else:
                # Try without password
                result = subprocess.run(
                    ['ssh', '-o', 'BatchMode=yes', 
                     '-o', 'StrictHostKeyChecking=no', 
                     '-o', 'UserKnownHostsFile=/dev/null',
                     f'{machine_user}@{machine_ip}', command],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
            
            # Return the result
            if result.returncode == 0:
                directories = result.stdout.strip().split('\n')
                
                # Filter out the current directory from the list
                directories = [d for d in directories if d != current_path]
                
                # Check for parent directory
                parent_path = ""
                if current_path != "~" and current_path != "/":
                    parent_path = os.path.dirname(current_path)
                    if not parent_path:
                        parent_path = "/"
                
                return jsonify({
                    'success': True,
                    'directories': directories,
                    'current_path': current_path,
                    'parent_path': parent_path
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to list directories',
                    'error': result.stderr.strip()
                })
                
        except subprocess.TimeoutExpired:
            return jsonify({'success': False, 'message': 'Command execution timed out'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid request method'})

@app.route('/tab-completion', methods=['POST'])
def tab_completion():
    """
    Handle tab completion requests for the terminal.
    This function executes a command on the remote machine to get possible completions.
    """
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            command = request.form.get('command', '')
            current_dir = request.form.get('current_directory', '~')
            
            if not machine_ip or not machine_user:
                return jsonify({
                    "success": False,
                    "message": "Missing required parameters"
                })
            
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
            
            # Get the token for completion
            tokens = command.strip().split()
            completion_token = ''
            
            if command.endswith(' '):
                # If command ends with space, we complete from the current directory
                completion_token = ''
            elif len(tokens) > 0:
                # Get the last token for completion
                completion_token = tokens[-1]
                
            # Create a bash script that uses compgen to get completions
            completion_script = f'''
            cd {current_dir} 2>/dev/null || cd ~
            
            # If the token contains a slash, we need to complete a path
            if [[ "{completion_token}" == *"/"* ]]; then
                # Get the directory part and the file part
                dir_part=$(dirname "{completion_token}")
                file_part=$(basename "{completion_token}")
                
                # Handle relative path
                if [[ ! "$dir_part" == /* && ! "$dir_part" == ~* ]]; then
                    # Current directory + dir_part
                    if [[ "$dir_part" == "." ]]; then
                        dir_part="$(pwd)"
                    else
                        dir_part="$(pwd)/$dir_part"
                    fi
                fi
                
                # List files in the directory matching the file part
                cd "$dir_part" 2>/dev/null && find . -maxdepth 1 -name "$file_part*" | cut -c3- | sort
                
                # Add trailing slash to directories
                cd "$dir_part" 2>/dev/null && find . -maxdepth 1 -type d -name "$file_part*" | cut -c3- | awk '{{ print $0"/" }}' | sort
            else
                # Use compgen for general command completion
                if [ -z "{completion_token}" ]; then
                    # No token, complete commands
                    compgen -c | sort | uniq
                else
                    # Try to complete commands first
                    compgen -c "{completion_token}" | sort | uniq
                    
                    # If in home directory, complete with tilde
                    if [[ "$(pwd)" == "$HOME" || "$(pwd)" == "/home/$USER" ]]; then
                        find . -maxdepth 1 -name "{completion_token}*" | cut -c3- | sort
                        # Add trailing slash to directories
                        find . -maxdepth 1 -type d -name "{completion_token}*" | cut -c3- | awk '{{ print $0"/" }}' | sort
                    else
                        # Complete files in the current directory
                        find . -maxdepth 1 -name "{completion_token}*" | cut -c3- | sort
                        # Add trailing slash to directories
                        find . -maxdepth 1 -type d -name "{completion_token}*" | cut -c3- | awk '{{ print $0"/" }}' | sort
                    fi
                fi
            fi
            '''
            
            # Use Paramiko for SSH connection
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            password = machine_passwords.get(machine_ip, '')
            
            try:
                ssh_client.connect(
                    hostname=machine_ip,
                    username=machine_user,
                    password=password,
                    timeout=10
                )
                
                # Execute the completion script
                stdin, stdout, stderr = ssh_client.exec_command(completion_script)
                
                # Get the completions
                completions = stdout.read().decode('utf-8').splitlines()
                error = stderr.read().decode('utf-8')
                
                # Close connection
                ssh_client.close()
                
                # Clean up the completions (remove duplicates and empty strings)
                completions = [c for c in completions if c.strip()]
                completions = list(dict.fromkeys(completions))  # Remove duplicates while preserving order
                
                # Return the completions
                return jsonify({
                    "success": True,
                    "completions": completions
                })
                
            except Exception as e:
                if ssh_client:
                    ssh_client.close()
                return jsonify({
                    "success": False,
                    "message": "Error during SSH connection",
                    "error": str(e)
                })
                
        except Exception as e:
            return jsonify({
                "success": False,
                "message": "Error processing tab completion",
                "error": str(e)
            })

@app.route('/list-remote-files', methods=['POST'])
def list_remote_files():
    """List files and directories on a remote machine"""
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            directory = request.form.get('directory', '.')
            
            if not machine_ip or not machine_user:
                return jsonify({
                    "success": False,
                    "message": "Missing required parameters"
                })
            
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
            
            # Connect to remote machine
            import paramiko
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=machine_ip,
                username=machine_user,
                password=password,
                timeout=10
            )
            
            # Get directory listing
            command = f'ls -la "{directory}"'
            stdin, stdout, stderr = ssh_client.exec_command(command)
            files_output = stdout.read().decode('utf-8')
            error_output = stderr.read().decode('utf-8')
            
            # Parse ls output to get file information
            files = []
            lines = files_output.strip().split('\n')
            
            # Skip the first line (total)
            if lines and lines[0].startswith('total'):
                lines = lines[1:]
                
            for line in lines:
                parts = line.split(None, 8)  # Split by whitespace, max 8 splits
                if len(parts) >= 9:
                    file_type = parts[0][0]  # First character of the permissions string
                    permissions = parts[0]
                    owner = parts[2]
                    group = parts[3]
                    size = parts[4]
                    date = ' '.join(parts[5:8])
                    name = parts[8]
                    
                    # Skip . and .. entries
                    if name == '.' or name == '..':
                        continue
                        
                    files.append({
                        'name': name,
                        'type': 'directory' if file_type == 'd' else 'file',
                        'permissions': permissions,
                        'owner': owner,
                        'group': group,
                        'size': size,
                        'date': date,
                        'path': os.path.join(directory, name).replace('\\', '/')
                    })
            
            # Get parent directory
            parent_dir = os.path.dirname(directory) if directory != '/' else '/'
            
            # Get current directory type using pwd
            stdin, stdout, stderr = ssh_client.exec_command('pwd')
            current_path = stdout.read().decode('utf-8').strip()
            
            # Get home directory 
            stdin, stdout, stderr = ssh_client.exec_command('echo $HOME')
            home_dir = stdout.read().decode('utf-8').strip()
            
            # Close connection
            ssh_client.close()
            
            return jsonify({
                "success": True,
                "files": files,
                "current_dir": directory,
                "parent_dir": parent_dir,
                "current_path": current_path,
                "home_dir": home_dir
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            })

@app.route('/get-remote-file', methods=['POST'])
def get_remote_file():
    """Get the contents of a file on a remote machine"""
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            file_path = request.form.get('file_path')
            
            if not machine_ip or not machine_user or not file_path:
                return jsonify({
                    "success": False,
                    "message": "Missing required parameters"
                })
            
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
            
            # Connect to remote machine
            import paramiko
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=machine_ip,
                username=machine_user,
                password=password,
                timeout=10
            )
            
            # Get file contents using cat
            command = f'cat "{file_path}"'
            stdin, stdout, stderr = ssh_client.exec_command(command)
            file_content = stdout.read().decode('utf-8', errors='replace')
            error_output = stderr.read().decode('utf-8')
            
            # Get file info
            command = f'file -b --mime-type "{file_path}"'
            stdin, stdout, stderr = ssh_client.exec_command(command)
            mime_type = stdout.read().decode('utf-8').strip()
            
            # Close connection
            ssh_client.close()
            
            # Check for errors
            if error_output:
                return jsonify({
                    "success": False,
                    "message": error_output
                })
            
            return jsonify({
                "success": True,
                "content": file_content,
                "path": file_path,
                "mime_type": mime_type,
                "filename": os.path.basename(file_path)
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            })

@app.route('/save-remote-file', methods=['POST'])
def save_remote_file():
    """Save contents to a file on a remote machine"""
    if request.method == 'POST':
        try:
            machine_ip = request.form.get('machine_ip')
            machine_user = request.form.get('machine_user')
            file_path = request.form.get('file_path')
            content = request.form.get('content')
            
            if not machine_ip or not machine_user or not file_path or content is None:
                return jsonify({
                    "success": False,
                    "message": "Missing required parameters"
                })
            
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
            
            # Connect to remote machine
            import paramiko
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(
                hostname=machine_ip,
                username=machine_user,
                password=password,
                timeout=10
            )
            
            # Create SFTP client
            sftp_client = ssh_client.open_sftp()
            
            # Write the content to the file
            with sftp_client.open(file_path, 'w') as f:
                f.write(content)
            
            # Close connection
            sftp_client.close()
            ssh_client.close()
            
            return jsonify({
                "success": True,
                "message": f"File {file_path} saved successfully"
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": str(e)
            })

@app.route('/database-stats')
def database_stats():
    """Separate page for displaying database statistics."""
    try:
        # Get all database information
        conn = create_connection()
        cursor = conn.cursor()
        databases = get_all_databases(cursor)
        cursor.close()
        conn.close()
        
        # Get disk space information - now returns a dictionary
        disk_info = get_disk_space()
        available_space = disk_info["free_space"]
        
        # Get total database size - without system databases
        try:
            _, total_db_size = get_total_database_size(include_system_dbs=False)
        except Exception as e:
            print(f"Error getting database size: {e}")
            total_db_size = "Unknown"
            
        # Get total database size - with system databases included
        try:
            _, total_with_system_db_size = get_total_database_size(include_system_dbs=True)
        except Exception as e:
            print(f"Error getting total size with system DBs: {e}")
            total_with_system_db_size = "Unknown"
        
        # Get actual MySQL directory size from disk
        mysql_dir_size = get_mysql_directory_size()
            
        return render_template('database_stats.html', 
                              databases=databases,
                              total_size=total_db_size,
                              total_with_system=total_with_system_db_size,
                              available_space=available_space,
                              disk_info=disk_info,
                              mysql_dir_size=mysql_dir_size)
                              
    except Exception as e:
        print(f"Error in database_stats: {e}")
        return f"Error loading database statistics: {str(e)}"

@app.route('/reset-conductance/<database>/<table_name>')
def reset_conductance(database, table_name):
    """Route to reset to original values by clearing conversion settings from the session."""
    # Clear conversion-related session variables
    session.pop('using_conductance', None)
    session.pop('conductance_params', None)
    session.pop('conductance_comparison', None)
    session.pop('using_linear_conversion', None)
    session.pop('linear_conversion_comparison', None)
    
    # Add a flash message for user feedback
    flash('Reset to original values successful. Original data will be used for plotting.', 'success')
    
    # Redirect back to the Choose Plot Function page
    return redirect(f'/view-plot/{database}/{table_name}/choose')

@app.route('/linear-conversion/<database>/<table_name>')
def linear_conversion(database, table_name):
    """Route to show custom linear conversion form."""
    try:
        # Get current linear conversion parameters
        from conductance_calculator import LINEAR_CONVERSION
        output_min = LINEAR_CONVERSION["output_min"]
        output_max = LINEAR_CONVERSION["output_max"]
        
        # Redirect to the custom linear conversion form
        return render_template('custom_linear_conversion.html', 
                              database=database, 
                              table_name=table_name,
                              output_min=output_min,
                              output_max=output_max)
        
    except Exception as e:
        print(f"Error showing linear conversion form: {str(e)}")
        flash(f"Error showing linear conversion form: {str(e)}", 'danger')
        return redirect(f'/view-plot/{database}/{table_name}/choose')

@app.route('/apply-custom-linear-conversion/<database>/<table_name>', methods=['POST'])
def apply_custom_linear_conversion(database, table_name):
    """Apply custom linear conversion with user-provided values."""
    try:
        # Get the form data
        output_min = float(request.form.get('output_min', 60))
        output_max = float(request.form.get('output_max', 170))
        
        # Update the global conversion parameters in the conductance calculator
        from conductance_calculator import update_linear_conversion_params
        update_linear_conversion_params(output_min, output_max)
        
        # Get original data for tables to create comparison
        table_names = table_name.split(',')
        comparison_tables = []
        
        for single_table in table_names:
            data_matrix, _ = get_full_table_data(single_table, database)
            comparison = get_unique_original_values_and_linear_conversion(data_matrix)
            comparison_tables.extend(comparison)
        
        # Remove duplicates and sort by original value
        unique_comparison = []
        seen = set()
        for item in comparison_tables:
            if item['original'] not in seen:
                unique_comparison.append(item)
                seen.add(item['original'])
        
        unique_comparison.sort(key=lambda x: x['original'])
        
        # Store settings in the session
        session['using_linear_conversion'] = True
        session['using_conductance'] = False  # Ensure conductance is turned off
        session['linear_conversion_comparison'] = unique_comparison
        
        # Add a flash message for user feedback
        flash(f'Linear conversion enabled. Values will be mapped from 0-63 to {output_min}-{output_max} range for plotting.', 'success')
        
        # Redirect back to the Choose Plot Function page
        return redirect(f'/view-plot/{database}/{table_name}/choose')
        
    except Exception as e:
        print(f"Error applying custom linear conversion: {str(e)}")
        flash(f"Error applying custom linear conversion: {str(e)}", 'danger')
        return redirect(f'/view-plot/{database}/{table_name}/choose')

@app.route('/top-schemas-by-size')
def top_schemas_by_size():
    """Display all schemas by size with search filtering."""
    try:
        # Check if user is logged in
        username = session.get('username')
        if not username:
            return redirect(url_for('login'))
            
        # Get schema sizes
        from db_operations import get_schema_sizes
        schema_sizes = get_schema_sizes()
        
        # Use all schemas instead of limiting to top 100
        all_schemas = schema_sizes
        
        # Calculate total size of all schemas for percentage calculation
        total_size_bytes = sum(schema['size_bytes'] for schema in schema_sizes)
        
        # Add percentage to each schema
        for schema in all_schemas:
            if total_size_bytes > 0:
                percentage = (schema['size_bytes'] / total_size_bytes) * 100
                schema['percentage'] = f"{percentage:.2f}%"
            else:
                schema['percentage'] = "0.00%"
        
        # Get disk space information
        disk_info = get_disk_space()
        
        # Get raw free space in bytes for comparison (10GB = 10 * 1024 * 1024 * 1024 bytes)
        disk_stats = shutil.disk_usage("/")  # Use parent directory to avoid permission issues
        free_space_gb = disk_stats.free / (1024 * 1024 * 1024)
        # Add free_space_gb to disk_info dictionary
        disk_info['free_space_gb'] = free_space_gb
        
        return render_template('top_schemas.html', 
                              schemas=all_schemas, 
                              username=username,
                              disk_info=disk_info,
                              total_schemas=len(schema_sizes),
                              total_size_bytes=total_size_bytes)
                              
    except Exception as e:
        print(f"Error in top_schemas_by_size: {e}")
        return f"Error loading top schemas by size: {str(e)}"

def save_bitmap_mask(database_name, mask_name, mask_array):
    """Save a bitmap mask to the database"""
    try:
        print(f"=== SAVING BITMAP MASK ===")
        print(f"Database: {database_name}")
        print(f"Mask name: {mask_name}")
        print(f"Mask shape: {mask_array.shape}")
        print(f"Mask type: {type(mask_array)}")
        
        # Use a global connection to store bitmap masks in a dedicated place
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Create bitmap_masks database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS bitmap_masks_db")
        cursor.execute("USE bitmap_masks_db")
        
        print(f"Using bitmap_masks_db database")
        
        # Check if bitmap_masks table exists
        cursor.execute("SHOW TABLES LIKE 'bitmap_masks'")
        table_exists = cursor.fetchone()
        
        print(f"Bitmap_masks table exists: {table_exists is not None}")
        
        if not table_exists:
            print("Creating bitmap_masks table...")
            cursor.execute("""
                CREATE TABLE bitmap_masks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    database_name VARCHAR(255) NOT NULL,
                    mask_name VARCHAR(255) NOT NULL,
                    dimensions VARCHAR(100) NOT NULL,
                    mask_data LONGTEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_mask (database_name, mask_name)
                )
            """)
            connection.commit()
            print("Table created successfully")
        
        # Convert numpy array to JSON string for storage
        import json
        mask_data_json = json.dumps(mask_array.tolist())
        dimensions = f"{mask_array.shape[0]}x{mask_array.shape[1]}"
        
        print(f"Mask dimensions: {dimensions}")
        print(f"JSON data length: {len(mask_data_json)}")
        
        # Insert or update the mask
        cursor.execute("""
            INSERT INTO bitmap_masks (database_name, mask_name, dimensions, mask_data)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            dimensions = VALUES(dimensions),
            mask_data = VALUES(mask_data),
            created_at = CURRENT_TIMESTAMP
        """, (database_name, mask_name, dimensions, mask_data_json))
        
        affected_rows = cursor.rowcount
        print(f"Database operation affected {affected_rows} rows")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"Bitmap mask '{mask_name}' saved successfully for database '{database_name}'")
        return True
        
    except Exception as e:
        import traceback
        print(f"Error saving bitmap mask: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def load_bitmap_mask(database_name, mask_name):
    """Load a bitmap mask from the database"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Use the bitmap_masks_db database
        cursor.execute("USE bitmap_masks_db")
        
        cursor.execute("""
            SELECT mask_data, dimensions 
            FROM bitmap_masks 
            WHERE database_name = %s AND mask_name = %s
        """, (database_name, mask_name))
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result:
            import json
            import numpy as np
            mask_data = json.loads(result[0])
            mask_array = np.array(mask_data)
            dimensions = result[1]
            print(f"Bitmap mask '{mask_name}' loaded successfully: {dimensions}")
            return mask_array, dimensions
        else:
            print(f"Bitmap mask '{mask_name}' not found for database '{database_name}'")
            return None, None
            
    except Exception as e:
        print(f"Error loading bitmap mask: {e}")
        return None, None

def validate_mask_dimensions(mask_array, data_matrix):
    """Validate that mask dimensions match data matrix dimensions"""
    if mask_array.shape != data_matrix.shape:
        return False, f"Dimension mismatch: mask is {mask_array.shape}, data is {data_matrix.shape}"
    return True, "Dimensions match"

def get_mysql_directory_size():
    """Get the actual disk space used by the MySQL data directory."""
    try:
        # Try to use the direct MySQL path first
        mysql_path = "/app/mysql"
        
        try:
            # Use du command to get the actual disk usage
            result = subprocess.run(['du', '-sh', mysql_path], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the output (format: "348G /app/mysql")
                output = result.stdout.strip()
                size_str = output.split()[0]
                
                # Convert to standardized format (e.g., "348.00 GB")
                if size_str.endswith('G'):
                    value = float(size_str[:-1])
                    formatted_size = f"{value:.2f} GB"
                elif size_str.endswith('M'):
                    value = float(size_str[:-1])
                    formatted_size = f"{value:.2f} MB"
                elif size_str.endswith('T'):
                    value = float(size_str[:-1])
                    formatted_size = f"{value:.2f} TB"
                else:
                    formatted_size = size_str
                    
                return formatted_size
        except (PermissionError, subprocess.SubprocessError):
            # If direct access fails due to permissions, estimate MySQL size
            # based on disk usage (assumed to be ~80% of used space on /app)
            try:
                disk_stats = shutil.disk_usage("/")
                used_gb = disk_stats.used / (1024 * 1024 * 1024)
                # Estimate MySQL size as 80% of used space
                mysql_estimated_gb = used_gb * 0.8
                return f"{mysql_estimated_gb:.2f} GB (estimated)"
            except Exception as inner_err:
                print(f"Error estimating MySQL size: {inner_err}")
                return "340.00 GB (estimated)"
        
        # Fallback to default value if both methods fail
        return "340.00 GB (default)"
    except Exception as e:
        print(f"Error getting MySQL directory size: {e}")
        return "340.00 GB (default)"

def split_wide_table_concatenation(database, table_names, base_table_name, max_columns_per_table=500):
    """
    Handle extremely wide tables by splitting them into multiple tables with fewer columns.
    This is a last-resort approach when the row size limit cannot be overcome.
    
    Args:
        database: The database name
        table_names: List of table names to concatenate
        base_table_name: Base name for the output tables
        max_columns_per_table: Maximum columns per output table (default 500, increased from 50)
        
    Returns:
        Number of tables created
    """
    import pandas as pd
    import numpy as np
    import traceback
    from db_operations import create_long_running_connection, create_long_running_engine
    
    try:
        # Create database connection and engine
        connection = create_long_running_connection(database)
        cursor = connection.cursor()
        engine = create_long_running_engine(database)
        
        # Set MySQL optimization settings to help with wide tables
        cursor.execute("SET SESSION innodb_strict_mode=OFF")
        cursor.execute("SET SESSION sql_mode=''")
        # Removed: cursor.execute("SET SESSION innodb_fill_factor=70")
        # Removed: cursor.execute("SET SESSION max_allowed_packet=1073741824")  # 1GB
        # Removed: cursor.execute("SET SESSION innodb_large_prefix=ON")
        # Removed: cursor.execute("SET GLOBAL innodb_file_per_table=ON")
        connection.commit()
        
        # First, check if all tables have the same row count
        row_counts = {}
        for table_name in table_names:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                count = cursor.fetchone()[0]
                row_counts[table_name] = count
            except Exception as e:
                print(f"Error getting row count for table {table_name}: {e}")
                connection.close()
                raise Exception(f'Error accessing table {table_name}: {str(e)}')
        
        # Check that all tables have the same count
        if len(set(row_counts.values())) != 1:
            connection.close()
            raise Exception('Tables have different row counts. All tables must have the same number of rows.')
        
        # Get the row count for processing
        row_count = next(iter(row_counts.values()))
        
        # Get all columns from all tables
        all_tables_columns = []
        
        for table_name in table_names:
            cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
            columns = [col[0] for col in cursor.fetchall()]
            for col in columns:
                all_tables_columns.append({
                    'table': table_name, 
                    'column': col
                })
        
        # Calculate how many tables we need
        num_tables = max(1, len(all_tables_columns) // max_columns_per_table + 
                          (1 if len(all_tables_columns) % max_columns_per_table > 0 else 0))
        
        print(f"Splitting {len(all_tables_columns)} columns across {num_tables} tables")
        
        # Split columns into groups
        column_groups = []
        for i in range(0, len(all_tables_columns), max_columns_per_table):
            end_idx = min(i + max_columns_per_table, len(all_tables_columns))
            column_groups.append(all_tables_columns[i:end_idx])
        
        # Create each table with its group of columns
        for table_idx, columns_group in enumerate(column_groups):
            # Generate a unique table name for this group
            output_table_name = f"{base_table_name}_{table_idx+1}"
            
            # Check if table already exists
            cursor.execute(f"SHOW TABLES LIKE '{output_table_name}'")
            if cursor.fetchone():
                cursor.execute(f"DROP TABLE `{output_table_name}`")
            
            # Create the table with TEXT columns to avoid row size limit issues
            create_sql = f"CREATE TABLE `{output_table_name}` ("
            col_defs = []
            
            # Add an index column to maintain the original row ordering
            col_defs.append("`row_idx` INT NOT NULL PRIMARY KEY")
            
            # Add all the columns for this group
            for col_info in columns_group:
                col_name = f"{col_info['table']}_{col_info['column']}"
                # Use TEXT instead of VARCHAR(255) to store data outside the row
                # This prevents row size limit issues with wide tables
                col_defs.append(f"`{col_name}` TEXT")
            
            create_sql += ", ".join(col_defs)
            create_sql += ") ENGINE=InnoDB ROW_FORMAT=DYNAMIC"
            
            # Add specific table options to handle row size limits
            cursor.execute("SET SESSION innodb_strict_mode=OFF")
            # Removed: cursor.execute("SET GLOBAL innodb_file_per_table=ON")
            
            # Execute the CREATE TABLE statement
            cursor.execute(create_sql)
            connection.commit()
            
            print(f"Created table {output_table_name} with {len(columns_group)} columns")
            
            # Now populate the table with data
            # First create a temporary dataframe to hold this batch of data
            df_columns = ['row_idx'] + [f"{col_info['table']}_{col_info['column']}" for col_info in columns_group]
            temp_df = pd.DataFrame(index=range(row_count), columns=df_columns)
            temp_df['row_idx'] = range(1, row_count + 1)  # 1-based index
            
            # Load the data for each column
            for col_info in columns_group:
                source_table = col_info['table']
                source_col = col_info['column']
                dest_col = f"{source_table}_{source_col}"
                
                # Fetch the data
                cursor.execute(f"SELECT `{source_col}` FROM `{source_table}`")
                rows = cursor.fetchall()
                
                # Add to the dataframe
                temp_df[dest_col] = [row[0] for row in rows]
            
            # Insert the data in chunks
            CHUNK_SIZE = 1000
            for start_idx in range(0, len(temp_df), CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, len(temp_df))
                chunk = temp_df.iloc[start_idx:end_idx]
                
                # Build insert SQL
                placeholders = ", ".join(["%s"] * len(df_columns))
                insert_sql = f"INSERT INTO `{output_table_name}` (`{'`, `'.join(df_columns)}`) VALUES ({placeholders})"
                
                # Insert each row
                values = []
                for _, row in chunk.iterrows():
                    values.append(tuple(row))
                    
                cursor.executemany(insert_sql, values)
                connection.commit()
                
                print(f"Inserted rows {start_idx} to {end_idx} into {output_table_name}")
        
        # Close connection
        cursor.close()
        connection.close()
        
        return num_tables
        
    except Exception as e:
        print(f"Error in split_wide_table_concatenation: {e}")
        traceback.print_exc()
        raise e

@app.route('/plot-selected', methods=['POST'])
def plot_selected():
    """
    Handle POST requests for plotting multiple tables.
    This avoids URL length limitations by using POST instead of GET.
    """
    database = request.form.get('database')
    plot_function = request.form.get('plot_function', 'None')
    table_names = request.form.getlist('tableNames')
    
    if not database or not table_names:
        return 'Missing required information', 400
    
    # Store table names in session to avoid URL length issues
    session['plot_tables'] = table_names
    
    # Redirect to view_plot with a special parameter indicating to use session data
    return redirect(url_for('view_plot', database=database, table_name='from_session', plot_function=plot_function))

@app.route('/process-plot-form', methods=['POST'])
def process_plot_form():
    """
    Handle POST form submission from input_form_generate_plot.html.
    This avoids URL length limitations by using POST instead of including table names in URL.
    """
    database = request.form.get('database')
    table_name = request.form.get('table_name')
    plot_function = request.form.get('plot_function')
    
    if not all([database, table_name, plot_function]):
        return 'Missing required parameters', 400
    
    # Generate plot form data based on the form
    if plot_function == "generate_plot":
        form_data = get_form_data_generate_plot(request.form)
        
        # Server-side validation for color_group_keywords
        if form_data.get('color_group_keywords'):
            table_names_list = table_name.split(',')
            keywords = form_data['color_group_keywords']
            conflict_messages = []
            for tn in table_names_list:
                tn_trimmed = tn.strip()
                matches = [kw for kw in keywords if kw in tn_trimmed]
                if len(matches) > 1:
                    conflict_messages.append(f"Table '{tn_trimmed}' matches multiple keywords: {', '.join(matches)}.")
            
            if conflict_messages:
                for msg in conflict_messages:
                    flash(msg, 'danger')
                flash("Please ensure each table name matches at most one keyword, or remove/adjust keywords.", 'danger')
                
                # Re-render the input form with existing context
                using_conductance = session.get('using_conductance', False)
                conductance_comparison = session.get('conductance_comparison', None)
                using_linear_conversion = session.get('using_linear_conversion', False)
                linear_conversion_comparison = session.get('linear_conversion_comparison', None)
                from conductance_calculator import LINEAR_CONVERSION
                linear_min = LINEAR_CONVERSION["output_min"]
                linear_max = LINEAR_CONVERSION["output_max"]
                
                return render_template('input_form_generate_plot.html', 
                                      database=database, 
                                      table_name=table_name, 
                                      plot_function=plot_function,
                                      using_conductance=using_conductance,
                                      conductance_comparison=conductance_comparison,
                                      using_linear_conversion=using_linear_conversion,
                                      linear_conversion_comparison=linear_conversion_comparison,
                                      linear_min=linear_min,
                                      linear_max=linear_max)
    else:
        return 'Invalid plot function', 400
    
    # Check if we should use conductance values
    using_conductance = session.get('using_conductance', False)
    if using_conductance:
        form_data['using_conductance'] = True
        form_data['conductance_params'] = session.get('conductance_params', {})
    
    # Check if we should use linear conversion
    using_linear_conversion = session.get('using_linear_conversion', False)
    if using_linear_conversion:
        form_data['using_linear_conversion'] = True
    
    # Convert form data to JSON
    form_data_json = json.dumps(form_data)
    
    # Store in session for retrieval in render_plot
    session['plot_form_data'] = form_data_json
    
    # Store table names in session (splitting if needed)
    table_names = table_name.split(',')
    session['plot_tables'] = table_names
    
    # Redirect to render-plot with a special parameter indicating to use session data
    return redirect(f"/render-plot/{database}/from_session/{plot_function}")

@app.route('/list-all-bitmap-masks')
def list_all_bitmap_masks():
    """List all bitmap masks in the database"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Check if bitmap_masks_db database exists
        cursor.execute("SHOW DATABASES LIKE 'bitmap_masks_db'")
        db_exists = cursor.fetchone()
        
        if not db_exists:
            return "<h3>No bitmap_masks_db database found</h3><p>The database will be created when you save your first bitmap mask.</p>"
        
        # Use the bitmap_masks_db database
        cursor.execute("USE bitmap_masks_db")
        
        # Check if bitmap_masks table exists
        cursor.execute("SHOW TABLES LIKE 'bitmap_masks'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            return "<h3>No bitmap_masks table found in bitmap_masks_db</h3><p>The table will be created when you save your first bitmap mask.</p>"
        
        # Get all masks
        cursor.execute("""
            SELECT id, database_name, mask_name, dimensions, created_at 
            FROM bitmap_masks 
            ORDER BY created_at DESC
        """)
        
        masks = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not masks:
            return "<h3>No bitmap masks found</h3><p>Create your first bitmap mask by using the Data Range Filter with 'Generate and save bitmap mask' checked.</p>"
        
        result = "<h3>All Bitmap Masks:</h3><ul>"
        for mask in masks:
            result += f"<li><strong>{mask[2]}</strong> (Database: {mask[1]}, Dimensions: {mask[3]}, Created: {mask[4]})</li>"
        result += "</ul>"
        
        return result
        
    except Exception as e:
        return f"Error checking bitmap masks: {str(e)}"

@app.route('/debug-bitmap-masks/<database>')
def debug_bitmap_masks(database):
    """Debug route to check bitmap masks in the database"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Check if bitmap_masks table exists
        cursor.execute("SHOW TABLES LIKE 'bitmap_masks'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            return f"bitmap_masks table does not exist for database {database}"
        
        # Get all masks for this database
        cursor.execute("""
            SELECT id, database_name, mask_name, dimensions, created_at 
            FROM bitmap_masks 
            WHERE database_name = %s 
            ORDER BY created_at DESC
        """, (database,))
        
        masks = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not masks:
            return f"No bitmap masks found for database {database}"
        
        result = f"<h3>Bitmap masks for database '{database}':</h3><ul>"
        for mask in masks:
            result += f"<li>ID: {mask[0]}, Name: {mask[1]}, Database: {mask[2]}, Dimensions: {mask[3]}, Created: {mask[4]}</li>"
        result += "</ul>"
        
        return result
        
    except Exception as e:
        return f"Error checking bitmap masks: {str(e)}"

@app.route('/get-bitmap-masks', methods=['POST'])
def get_bitmap_masks():
    """Get available bitmap masks for a database"""
    try:
        data = request.get_json()
        database = data.get('database')
        
        if not database:
            return jsonify({'error': 'Database parameter required'}), 400
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Check if bitmap_masks_db database exists
        cursor.execute("SHOW DATABASES LIKE 'bitmap_masks_db'")
        db_exists = cursor.fetchone()
        
        if not db_exists:
            cursor.close()
            connection.close()
            return jsonify({'masks': []})
        
        # Use the bitmap_masks_db database
        cursor.execute("USE bitmap_masks_db")
        
        # Check if bitmap_masks table exists
        cursor.execute("SHOW TABLES LIKE 'bitmap_masks'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            cursor.close()
            connection.close()
            return jsonify({'masks': []})
        
        # Get available masks for this database
        cursor.execute("""
            SELECT mask_name, dimensions 
            FROM bitmap_masks 
            WHERE database_name = %s 
            ORDER BY created_at DESC
        """, (database,))
        
        masks = []
        for row in cursor.fetchall():
            masks.append({
                'name': row[0],
                'dimensions': row[1]
            })
        
        cursor.close()
        connection.close()
        
        return jsonify({'masks': masks})
        
    except Exception as e:
        print(f"Error getting bitmap masks: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request"}), 400
    if request.files['file'].filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    file = request.files['file']
    
    import hashlib
    hash_object = hashlib.sha256()
    print('key is', request.form['key'].encode('utf-8'))
    hash_object.update(request.form['key'].encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    filepath = os.path.join(UPLOAD_FOLDER, hash_hex, file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    return jsonify({'message': f'Saved to {filepath}'})

@app.route('/uploaded_files/<hash_hex>/<filename>')
def serve_uploaded_file(hash_hex, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, hash_hex), filename)

@app.route('/recent_tables', methods=['POST', 'GET'])
def recent_tables():
    import hashlib

    dic = {}
    imageMap = defaultdict(list)
    engine = create_db_engine('Recent_Tables')

    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM records"))
        tables = [row[0] for row in result][::-1]

    for t in tables:
        hash_object = hashlib.sha256()
        hash_object.update(t.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        dic[t] = hash_hex
    
        filepath = os.path.join(UPLOAD_FOLDER, hash_hex)
        if os.path.exists(filepath):
            imageMap[t] = os.listdir(filepath)
        else:
            imageMap[t] = []

    return render_template('recent_plot.html', tables= tables, imageMap = imageMap, dic = dic)

def parse_macro_filter(macro_str):
    """
    Parse macro filter string that supports:
    - Single values: "1", "2", "3"
    - Ranges: "4~6" (includes 4, 5, 6)
    - Combinations: "1,4~6,8,10~11"
    Returns a list of integer values
    """
    if not macro_str or macro_str.strip() == '':
        return []
    
    macro_values = []
    
    # Split by comma to get individual parts
    parts = [part.strip() for part in macro_str.split(',')]
    
    for part in parts:
        if '~' in part:
            # Handle range (e.g., "4~6")
            try:
                start, end = part.split('~')
                start_val = int(start.strip())
                end_val = int(end.strip())
                macro_values.extend(range(start_val, end_val + 1))
            except (ValueError, IndexError):
                # If range parsing fails, try as single value
                try:
                    macro_values.append(int(part))
                except ValueError:
                    pass
        else:
            # Handle single value
            try:
                macro_values.append(int(part))
            except ValueError:
                pass
    
    # Remove duplicates and sort
    return sorted(list(set(macro_values)))

@app.route('/gui-rwb-analysis')
def gui_rwb_analysis():
    """Display the GUI RWB analysis form"""
    try:
        # Get all column names from the RWB database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='rwb'
        )
        cursor = connection.cursor()
        
        # Get column names
        cursor.execute("SHOW COLUMNS FROM rwb_db_3")
        columns = [col[0] for col in cursor.fetchall()]
        
        cursor.close()
        connection.close()
        
        return render_template('rwb_analysis_form.html', available_columns=columns)
    
    except Exception as e:
        flash(f'Error loading column information: {str(e)}', 'error')
        return render_template('rwb_analysis_form.html', available_columns=[])

@app.route('/process-rwb-analysis', methods=['POST'])
def process_rwb_analysis():
    """Process the RWB analysis form and generate results"""
    # Extract form data first so we can use it in error handling
    regex = request.form.get('regex', '').strip()
    regex_col = request.form.get('regex_col', 'DIE_ID')
    
    # If regex is empty, don't apply regex filtering
    if not regex:
        regex = False
    date = request.form.get('date', '')
    skip_rwb_2 = 'skip_rwb_2' in request.form
    
    # Get filter values - empty means no filter applied
    test_name_filter = request.form.get('test_name_filter', '').strip()
    macro_filter_str = request.form.get('macro_filter', '').strip()
    datetime_filter = request.form.get('datetime_filter', '').strip()
    
    # Parse macro filter only if provided - can be single values, ranges, or combinations
    macro_values = parse_macro_filter(macro_filter_str) if macro_filter_str else None
    
    overlay_col = request.form.get('overlay_col', 'IO')
    legend = 'legend' in request.form
    plot_title = request.form.get('plot_title', 'DOE21: BLREF_CAL by IO')
    
    groupby_cols = request.form.get('groupby_cols', '').strip()
    # If empty, pass None to the function for no grouping
    if groupby_cols:
        groupby_cols_list = [col.strip() for col in groupby_cols.split(',') if col.strip()]
        # If all columns were empty strings after splitting, treat as no grouping
        if not groupby_cols_list:
            groupby_cols_list = None
            groupby_cols = 'None (no grouping)'
    else:
        groupby_cols_list = None
        groupby_cols = 'None (no grouping)'
    
    # Prepare default analysis summary for error cases
    analysis_summary = {
        'total_records': 0,
        'filtered_records': 0,
        'regex_pattern': regex if regex != False else 'None (no regex)',
        'regex_col': regex_col,
        'plot_title': plot_title,
        'test_name_filter': test_name_filter or 'None (no filter)',
        'macro_filter': macro_filter_str or 'None (no filter)',
        'datetime_filter': datetime_filter or 'None (no filter)',
        'groupby_cols': groupby_cols,
        'overlay_col': overlay_col
    }
    
    try:
        # Import necessary modules
        import sys
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Add postprocess directory to path
        postprocess_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postprocess')
        sys.path.insert(0, postprocess_dir)
        
        # Import the core functions
        from core_post_processing_functions import rwb_fetch_data, rwb_groupby_level_nqplot, rwb_calc_io_mean
        
        # Step 1: Fetch data
        print("Fetching RWB data...")
        date_param = date if date else None
        rwb_pull = rwb_fetch_data(regex=regex, regex_col=regex_col, date=date_param, skip_rwb_2=skip_rwb_2)
        
        analysis_summary['total_records'] = len(rwb_pull)
        
        if rwb_pull.empty:
            return render_template('rwb_plots.html', 
                                 analysis_summary=analysis_summary,
                                 error_message="No data found with the specified parameters.")
        
        # Step 2: Apply filters
        print("Applying filters...")
        
        # Start with all data
        rwb_plot = rwb_pull.copy()
        
        # Apply TEST_NAME filter if specified (non-empty)
        if test_name_filter:
            print(f"Applying TEST_NAME filter: {test_name_filter}")
            rwb_plot = rwb_plot.loc[rwb_plot.TEST_NAME.str.contains(test_name_filter, na=False)]
        
        # Apply MACRO filter if specified (non-empty)
        if macro_values:
            print(f"Applying MACRO filter: {macro_values}")
            rwb_plot = rwb_plot.loc[rwb_plot.MACRO.isin(macro_values)]
        
        # Apply DATETIME filter if specified (non-empty)
        if datetime_filter:
            print(f"Applying DATETIME filter: {datetime_filter}")
            rwb_plot = rwb_plot.loc[rwb_plot.TEST_START_DATETIME.str.contains(datetime_filter, na=False)]
        
        analysis_summary['filtered_records'] = len(rwb_plot)
        
        if rwb_plot.empty:
            return render_template('rwb_plots.html', 
                                 analysis_summary=analysis_summary,
                                 error_message="No data found after applying filters.")
        
        # Step 3: Generate plot
        print("Generating plot...")
        plt.figure(figsize=(10, 6))
        
        # Configure matplotlib to not display plots
        plt.ioff()
        
        # Call the plotting function
        rwb_groupby_level_nqplot(rwb_plot, overlay_col=overlay_col, legend=legend, title=plot_title)
        
        # Convert plot to HTML
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        plot_html = f'<img src="data:image/png;base64,{img_string}" class="img-fluid" alt="RWB Plot"/>'
        
        # Step 4: Calculate mean
        print("Calculating mean...")
        rwb_mean = rwb_calc_io_mean(rwb_plot, groupby_cols=groupby_cols_list)
        
        # Step 5: Extract specific PPM results
        ppm_columns = ['TEST_NAME', 'LEVEL_01_XPOINT_PPM', 'LEVEL_12_XPOINT_PPM', 'LEVEL_23_XPOINT_PPM']
        available_columns = [col for col in ppm_columns if col in rwb_mean.columns]
        
        if available_columns:
            ppm_results = rwb_mean[available_columns]
        else:
            ppm_results = None
        
        # Convert DataFrames to safe format for Jinja2 template evaluation
        # Convert to dictionary format to avoid pandas DataFrame boolean evaluation issues
        if rwb_mean is not None and len(rwb_mean) > 0:
            mean_results_to_pass = {
                'data': rwb_mean.to_dict(orient='records'),
                'columns': rwb_mean.columns.tolist(),
                'has_data': True
            }
        else:
            mean_results_to_pass = None
            
        if ppm_results is not None and len(ppm_results) > 0:
            ppm_results_to_pass = {
                'data': ppm_results.to_dict(orient='records'),
                'columns': ppm_results.columns.tolist(),
                'has_data': True
            }
        else:
            ppm_results_to_pass = None
        
        return render_template('rwb_plots.html', 
                             plot_html=plot_html,
                             mean_results=mean_results_to_pass,
                             ppm_results=ppm_results_to_pass,
                             analysis_summary=analysis_summary,
                             success_message="Analysis completed successfully!")
                             
    except Exception as e:
        import traceback
        error_msg = f"Error during analysis: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        
        # Check if it's a missing dependency error
        if "ModuleNotFoundError" in str(e) and "anyio" in str(e):
            error_msg = "Missing required dependency 'anyio'. Please install it by running: pip install anyio"
        
        return render_template('rwb_plots.html', 
                             analysis_summary=analysis_summary,
                             error_message=error_msg)

@app.route('/get-rwb-column-values/<column_name>')
def get_rwb_column_values(column_name):
    """Get unique values for a specific RWB column"""
    try:
        # Create database connection
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='rwb'
        )
        cursor = connection.cursor()
        
        # First, validate the column exists
        cursor.execute("SHOW COLUMNS FROM rwb_db_3")
        columns = [col[0] for col in cursor.fetchall()]
        
        if column_name not in columns:
            cursor.close()
            connection.close()
            return jsonify({'error': f'Column "{column_name}" not found'}), 400
        
        # Get unique values for the column (limit to prevent memory issues)
        # Use DISTINCT and LIMIT to get a reasonable sample of values
        query = f"SELECT DISTINCT `{column_name}` FROM rwb_db_3 WHERE `{column_name}` IS NOT NULL ORDER BY `{column_name}` LIMIT 500"
        cursor.execute(query)
        
        values = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'values': values,
            'count': len(values),
            'column_name': column_name
        })
        
    except mysql.connector.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error getting column values: {str(e)}'}), 500

@app.route('/get-rwb-filtered-values', methods=['POST'])
def get_rwb_filtered_values():
    """Get unique values for a column filtered by other column conditions"""
    try:
        data = request.get_json()
        column_name = data.get('column_name', '').strip()
        filters = data.get('filters', {})
        
        if not column_name:
            return jsonify({'error': 'Column name is required'}), 400
        
        # Create database connection
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='rwb'
        )
        cursor = connection.cursor()
        
        # First, validate the column exists
        cursor.execute("SHOW COLUMNS FROM rwb_db_3")
        columns = [col[0] for col in cursor.fetchall()]
        
        if column_name not in columns:
            cursor.close()
            connection.close()
            return jsonify({'error': f'Column "{column_name}" not found'}), 400
        
        # Build WHERE clause based on filters
        where_conditions = [f"`{column_name}` IS NOT NULL"]
        params = []
        
        for filter_col, filter_value in filters.items():
            if filter_value and filter_value.strip() and filter_col in columns:
                if filter_col == 'MACRO' and (',' in filter_value or '~' in filter_value):
                    # Handle MACRO filter with ranges/multiple values
                    macro_values = parse_macro_filter(filter_value)
                    if macro_values:
                        placeholders = ','.join(['%s'] * len(macro_values))
                        where_conditions.append(f"`{filter_col}` IN ({placeholders})")
                        params.extend(macro_values)
                else:
                    # Handle other filters with string contains
                    where_conditions.append(f"`{filter_col}` LIKE %s")
                    params.append(f"%{filter_value}%")
        
        # Build and execute query
        where_clause = " AND ".join(where_conditions)
        query = f"SELECT DISTINCT `{column_name}` FROM rwb_db_3 WHERE {where_clause} ORDER BY `{column_name}` LIMIT 500"
        
        cursor.execute(query, params)
        values = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'values': values,
            'count': len(values),
            'column_name': column_name,
            'filters_applied': filters
        })
        
    except mysql.connector.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error getting filtered column values: {str(e)}'}), 500

@app.route('/get-rwb-multiple-regex-filtered-values', methods=['POST'])
def get_rwb_multiple_regex_filtered_values():
    """Get unique values for a column filtered by multiple regex conditions"""
    try:
        data = request.get_json()
        target_column = data.get('target_column', '').strip()
        filters = data.get('filters', {})
        
        if not target_column:
            return jsonify({'error': 'Target column is required'}), 400
        
        # Create database connection
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='rwb'
        )
        cursor = connection.cursor()
        
        # First, validate the target column exists
        cursor.execute("SHOW COLUMNS FROM rwb_db_3")
        columns = [col[0] for col in cursor.fetchall()]
        
        if target_column not in columns:
            cursor.close()
            connection.close()
            return jsonify({'error': f'Column "{target_column}" not found'}), 400
        
        # Build WHERE clause based on multiple regex filters
        where_conditions = [f"`{target_column}` IS NOT NULL"]
        params = []
        
        # Process cascaded filters (regex_1_column, regex_1_pattern, regex_2_column, etc.)
        filter_pairs = {}
        for key, value in filters.items():
            if key.endswith('_column'):
                pair_id = key.replace('_column', '')
                if pair_id not in filter_pairs:
                    filter_pairs[pair_id] = {}
                filter_pairs[pair_id]['column'] = value
            elif key.endswith('_pattern'):
                pair_id = key.replace('_pattern', '')
                if pair_id not in filter_pairs:
                    filter_pairs[pair_id] = {}
                filter_pairs[pair_id]['pattern'] = value
        
        # Apply each filter pair
        for pair_id, pair_data in filter_pairs.items():
            filter_col = pair_data.get('column', '').strip()
            filter_value = pair_data.get('pattern', '').strip()
            
            if filter_col and filter_value and filter_col in columns:
                if filter_col == 'MACRO' and (',' in filter_value or '~' in filter_value):
                    # Handle MACRO filter with ranges/multiple values
                    macro_values = parse_macro_filter(filter_value)
                    if macro_values:
                        placeholders = ','.join(['%s'] * len(macro_values))
                        where_conditions.append(f"`{filter_col}` IN ({placeholders})")
                        params.extend(macro_values)
                else:
                    # Handle other filters with string contains (regex-like)
                    where_conditions.append(f"`{filter_col}` LIKE %s")
                    params.append(f"%{filter_value}%")
        
        # Build and execute query
        where_clause = " AND ".join(where_conditions)
        query = f"SELECT DISTINCT `{target_column}` FROM rwb_db_3 WHERE {where_clause} ORDER BY `{target_column}` LIMIT 500"
        
        cursor.execute(query, params)
        values = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'values': values,
            'count': len(values),
            'column_name': target_column,
            'filters_applied': filters
        })
        
    except mysql.connector.Error as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error getting filtered column values: {str(e)}'}), 500

