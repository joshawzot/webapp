# route_handlers.py
# Note: File Explorer functionality has been removed from this application
# The following routes were removed:
# - /file-explorer
# - /api/list-directory
# - /api/read-file
# - /api/save-file
# - /api/create-file
# - /api/create-directory

from run import app, cache, redis_client
from db_operations import *
from tools_for_plots import get_full_table_data, plot_individual_points_map  # Add plot_individual_points_map to the import
from flask_caching import Cache

# Standard library imports
import os, base64, json, time
from io import BytesIO

# External libraries
import pandas as pd
import mysql.connector
from flask import Flask, request, make_response, redirect, url_for, session, send_file, render_template, render_template_string, jsonify
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
#from generate_plot_ber_by_bls import generate_plot_ber_by_bls
from generate_plot_read_stability import generate_plot_read_stability

#from flask_caching import Cache
#cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Add this near the top of the file, with the other imports
import json
from datetime import datetime

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
        
        # Keep only the 5 most recent visits
        recent_visits = recent_visits[:5]
        
        # Save back to Redis
        redis_client.set('recent_folder_visits', json.dumps(recent_visits))
        
        return True
    except Exception as e:
        print(f"Error recording folder visit: {e}")
        return False

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
                
            cursor.close()
            conn.close()
            return render_template('home_page.html', databases=databases, username=username, recent_visits=recent_visits)
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

    return render_template('list_tables.html', tables=tables, table_names=table_names, database=database, plot_function=plot_function)

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
    #"generate_plot_ber_by_bls": generate_plot_ber_by_bls,
    "generate_plot_read_stability": generate_plot_read_stability,
}

@app.route('/render-plot/<database>/<table_name>/<plot_function>')
def render_plot(database, table_name, plot_function):
    try:
        if 'username' not in session:
            return "User not logged in", 403

        # Parse form data
        form_data_json = request.args.get('form_data', '{}')
        try:
            form_data = json.loads(form_data_json)
        except json.JSONDecodeError:
            return "Error: Invalid form data", 400

        # Map plot function names to actual functions
        plot_functions = {
            'generate_plot': generate_plot,
            'generate_plot_read_stability': generate_plot_read_stability,
            #'generate_plot_ber_by_bls': generate_plot_ber_by_bls,
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
                (plot_data,
                 sorted_table_names,
                 sorted_table_names_100ppm,
                 sorted_table_names_200ppm,
                 sorted_table_names_500ppm,
                 sorted_table_names_1000ppm,
                 best_32,
                 best_32_with_io,
                 outlier_coordinates,
                 correlation_analysis,
                 cluster_map,
                 sigma_distances,
                 num_states,
                 table_names,
                 sigma_table,
                 sigma_points) = plot_function_impl(table_name.split(','), database, form_data)
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

            return render_template(
                'plot.html',
                plot_data=plot_data,
                sorted_table_names=sorted_table_names,
                sorted_table_names_100ppm=sorted_table_names_100ppm,
                sorted_table_names_200ppm=sorted_table_names_200ppm,
                sorted_table_names_500ppm=sorted_table_names_500ppm,
                sorted_table_names_1000ppm=sorted_table_names_1000ppm,
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
                sigma_points=sigma_points
            )

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

@app.route('/view-plot/<database>/<table_name>/<plot_function>', methods=['GET', 'POST'])
def view_plot(database, table_name, plot_function):
    print("view_plot")
    if request.method == "POST":
        print("POST:::::::::::::::::::::::::::::::::")
        plot_function_choice = request.form.get('plot_choice')
        if plot_function_choice:
            plot_function = plot_function_choice
            if plot_function in generate_plot_functions:
                if plot_function == "generate_plot":
                    return render_template('input_form_generate_plot.html', database=database, table_name=table_name, plot_function=plot_function)
                elif plot_function == "generate_plot_read_stability":
                    return render_template('input_form_generate_plot_read_stability.html', database=database, table_name=table_name, plot_function=plot_function)
                    #elif plot_function == "generate_plot_ber_by_bls":
                        #return render_template('input_form_generate_plot_ber_by_bls.html', database=database, table_name=table_name, plot_function=plot_function)
            else:
                return f"Invalid plot function selection", 400

        if plot_function:
            print(f"plot_function: {plot_function}")
            # if plot_function in generate_plot_functions:
            #     form_data = get_form_data_generate_plot(request.form)
            #     form_data_json = json.dumps(form_data)

            if plot_function in ["generate_plot", "generate_plot_read_stability", '''"generate_plot_ber_by_bls"''']:
                # Determine the appropriate function to call based on plot_function
                if plot_function == "generate_plot":
                    form_data = get_form_data_generate_plot(request.form)
                elif plot_function == "generate_plot_read_stability":
                    form_data = get_form_data_generate_plot_read_stability(request.form)
                    #elif plot_function == "generate_plot_ber_by_bls":
                        #form_data = get_form_data_generate_plot_ber_by_bls(request.form)
        
                # Convert form data to JSON
                form_data_json = json.dumps(form_data)

                # Redirect with the form data in the query string
                return redirect(f"/render-plot/{database}/{table_name}/{plot_function}?form_data={form_data_json}")
            else:
                return f"Invalid plot function selection", 400
        else:
            return f"Plot function not selected", 400
    else:
        table_names = table_name.split(',')
        print(table_names)
        print("GET:::::::::::::::::::::::::::::::::")
        return render_template('choose_plot_function_form.html')

from sqlalchemy import text
import pandas as pd

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
        tables = request.json['tables']
        connection = create_connection(database)
        cursor = connection.cursor()

        for table_name in tables:
            query = f"DROP TABLE `{table_name}`"
            cursor.execute(query)

        connection.commit()
        close_connection()

        return "Records deleted successfully", 200
    except mysql.connector.Error as err:
        return str(err), 400

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
    template_path = '/home/admin2/webapp_2/pptx_template/template.pptx'

    plots = request.json.get('plots', [])  # Retrieve the Base64 encoded images from the POST request

    prs = Presentation(template_path)  # Open the template PowerPoint file as the base for the new presentation
    
    # Retrieve the Base64 encoded images from the POST request
    plots = request.json.get('plots', [])
    
    for plot_data in plots:
        # Decode each Base64 image
        image_data = base64.b64decode(plot_data.split(",")[-1])
        # Open the image for analysis
        image = Image.open(BytesIO(image_data))
        
        # Choose a slide layout (6 is usually a blank slide)
        slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(slide_layout)
        
        # Remove all shapes (including text boxes) from the slide
        for shape in slide.shapes:
            sp = shape._element
            sp.getparent().remove(sp)
        
        # Get the image size
        img_width, img_height = image.size
        # Get the slide dimensions
        slide_width = prs.slide_width
        slide_height = prs.slide_height
        
        # Calculate the scaling factor to maintain aspect ratio
        ratio = min(slide_width / img_width, slide_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Center the image
        left = int((slide_width - new_width) / 2)
        top = int((slide_height - new_height) / 2)
        
        # Convert the image data back to a BytesIO object
        img_io = BytesIO(image_data)
        # Add the image to the slide
        slide.shapes.add_picture(img_io, left, top, width=new_width, height=new_height)
    
    # Prepare the presentation to be sent in the response
    pptx_io = BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    
    # Set up the response with the correct headers
    response = make_response(pptx_io.getvalue())
    response.headers.set('Content-Type', 'application/vnd.openxmlformats-officedocument.presentationml.presentation')
    response.headers.set('Content-Disposition', 'attachment; filename="Downloaded_Presentation.pptx"')
    
    return response

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
            new_columns.append(col_str)
    
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

    try:
        # Create a connection specifying the database
        connection = create_connection(database)
        cursor = connection.cursor()
        
        # List to hold reshaped arrays
        reshaped_arrays = []

        # Load the pattern file based on state_pattern
        pattern_files = {
            "1296x64_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_rowbar_4states.npy",
            "3x4_4states_debug": "/home/admin2/webapp_2/State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "/home/admin2/webapp_2/State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "/home/admin2/webapp_2/State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "/home/admin2/webapp_2/State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "/home/admin2/webapp_2/State_pattern_files/248x248_16states.npy",
            "248x1_1state": "/home/admin2/webapp_2/State_pattern_files/248x1_1state.npy"
        }

        file_path = pattern_files.get(state_pattern)
        if not file_path or not os.path.exists(file_path):
            print(f"DEBUG: Pattern file not found: {file_path}")
            return 'Invalid state pattern or file not found', 400

        # Load the pattern array and flatten it
        print(f"DEBUG: Loading pattern file from: {file_path}")
        pattern_array = np.load(file_path)
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

        for table_name in table_names:
            print(f"\nDEBUG: Processing table: {table_name}")
            query = f"SELECT * FROM `{table_name}`"
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"DEBUG: Fetched {len(rows)} rows from table")
            
            if len(rows) == 0:
                print(f"DEBUG: Warning - Empty table: {table_name}")
                continue
                
            columns = [desc[0] for desc in cursor.description]
            print(f"DEBUG: Column count: {len(columns)}")
            
            df = pd.DataFrame(rows, columns=columns)
            print(f"DEBUG: DataFrame shape: {df.shape}")
            
            # Check for missing values
            if df.isnull().values.any():
                print("DEBUG: Warning - DataFrame contains NaN values")
            
            # Convert DataFrame to numpy array
            arr = df.to_numpy()
            print(f"DEBUG: Array shape: {arr.shape}, dtype: {arr.dtype}")
            
            # Print a sample of the array
            if arr.size > 0:
                print(f"DEBUG: Array sample (first element): {arr.flat[0]}")
            
            # Flatten the array
            arr_flat = arr.flatten()
            print(f"DEBUG: Flattened array shape: {arr_flat.shape}, size: {arr_flat.size}")
            
            # Check for size mismatch before proceeding
            if arr_flat.size != pattern_flat.size:
                print(f"DEBUG: Size mismatch! arr_flat.size: {arr_flat.size}, pattern_flat.size: {pattern_flat.size}")
                return f'Pattern file ({pattern_flat.size} elements) and table data ({arr_flat.size} elements) dimensions do not match', 400

            # Get indices that would sort the pattern_flat
            print("DEBUG: Calculating argsort of pattern_flat...")
            pattern_indices = np.argsort(pattern_flat)
            print(f"DEBUG: Pattern indices shape: {pattern_indices.shape}")
            print(f"DEBUG: First 10 indices: {pattern_indices[:10]}")

            # Reorder arr_flat according to pattern_indices
            print("DEBUG: Reordering arr_flat according to pattern_indices...")
            try:
                arr_reordered = arr_flat[pattern_indices]
                print(f"DEBUG: Reordered array shape: {arr_reordered.shape}")
            except Exception as e:
                print(f"DEBUG: Error during reordering: {str(e)}")
                return f'Error during reordering: {str(e)}', 500

            # Reshape to (a*b, 1)
            print(f"DEBUG: Reshaping to ({a*b}, 1)...")
            try:
                arr_new = arr_reordered.reshape((a * b, 1))
                print(f"DEBUG: Reshaped array shape: {arr_new.shape}")
            except Exception as e:
                print(f"DEBUG: Error during reshaping: {str(e)}")
                return f'Error during reshaping: {str(e)}', 500

            # Append to list
            reshaped_arrays.append(arr_new)
            print(f"DEBUG: Successfully processed table: {table_name}")

        print(f"\nDEBUG: All tables processed. Reshaped arrays count: {len(reshaped_arrays)}")
        
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

            # Create the new table in the database using SQLAlchemy engine
            print(f"DEBUG: Saving data to new table: {new_table_name}")
            engine = create_db_engine(database)
            combined_df.to_sql(new_table_name, engine, if_exists='fail', index=False)
            print("DEBUG: Table saved successfully")

            # Clean up
            cursor.close()
            close_connection()

            # After successful merging, redirect to list_tables
            return redirect(url_for('list_tables', database=database))

        else:
            # Clean up
            cursor.close()
            close_connection()
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

    # Connect to databases
    source_conn = create_connection(source_db)
    target_conn = create_connection(target_db)

    try:
        # Get existing table names in the target database
        existing_tables = get_table_names(target_conn)

        # Find conflicts
        conflicts = set(table_names) & set(existing_tables)
        if conflicts:
            conflict_list = ', '.join(conflicts)
            return jsonify({'message': f'The following tables already exist in the target database: {conflict_list}'}), 400

        # Copy tables
        for table in table_names:
            # Fetch table data from the source database
            table_data = get_table_from_database(source_db, table)
            if table_data is None:
                return jsonify({'message': f'Failed to retrieve data for table {table}.'}), 500

            data_rows, columns = table_data

            # Create the table in the target database
            create_table(target_conn, table, data_rows, columns)

        return jsonify({'message': 'Tables copied successfully.'}), 200

    except Exception as e:
        print(f'Error copying tables: {e}')
        return jsonify({'message': 'An error occurred while copying tables.'}), 500

    finally:
        # Close database connections
        source_conn.close()
        target_conn.close()

@app.route('/concatenate_tables', methods=['POST'])
def concatenate_tables():
    data = request.json
    database = data.get('database')
    table_names = data.get('tableNames')
    new_table_name = data.get('newTableName')
    
    # Set a reasonable batch size for processing large tables
    BATCH_SIZE = 5  # Process 5 tables at a time

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
            return jsonify(success=False, message='A table with the new name already exists.')
        
        # Fetch data from each table and prepare for concatenation
        print(f"Preparing to concatenate {len(table_names)} tables")
        
        # Create a dictionary to store column name mappings for each table
        column_mappings = {}
        all_df_columns = []
        
        # Load data from tables and track column names
        for i, table_name in enumerate(table_names):
            # Get column information for each table
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns_info = cursor.fetchall()
            
            # Get column names and create prefixed versions
            original_columns = [col[0] for col in columns_info]
            prefixed_columns = [f"t{i}_{col}" for col in original_columns]
            
            # Store the mapping for this table
            column_mappings[table_name] = {
                'original': original_columns,
                'prefixed': prefixed_columns
            }
            
            # Add prefixed columns to the full list
            all_df_columns.extend(prefixed_columns)
        
        # Create the empty dataframe for concatenation
        print(f"Creating empty dataframe with {row_count} rows")
        
        # Efficiently build the dataframe by creating it just once with NaN values
        import numpy as np
        import pandas as pd
        empty_df = pd.DataFrame(np.nan, index=range(row_count), columns=all_df_columns)
        
        # Now fill the dataframe with data from each table
        for i, table_name in enumerate(table_names):
            print(f"Loading data from table {i+1}/{len(table_names)}: {table_name}")
            
            # Efficiently fetch all data at once
            cursor.execute(f"SELECT * FROM `{table_name}`")
            rows = cursor.fetchall()
            
            # Get original column names
            original_columns = column_mappings[table_name]['original']
            prefixed_columns = column_mappings[table_name]['prefixed']
            
            # Convert to dataframe
            table_df = pd.DataFrame(rows, columns=original_columns)
            
            # Efficiently copy data to the main dataframe
            for orig_col, pref_col in zip(original_columns, prefixed_columns):
                empty_df[pref_col] = table_df[orig_col].values
            
            # Clear memory
            del table_df
        
        # Clean up column names - use our rename function to ensure uniqueness
        # First, remove the temporary prefixes
        renamed_columns = {}
        for col in empty_df.columns:
            if col.startswith('t') and '_' in col:
                base_name = col.split('_', 1)[1]
                renamed_columns[col] = base_name
        
        # Now handle duplicate column names using our improved function
        seen = {}
        final_column_mapping = {}
        
        for old_col, base_name in renamed_columns.items():
            # Check if this base name has been seen before
            if base_name in seen:
                seen[base_name] += 1
                new_col = f"{base_name}_{seen[base_name]}"
                # Ensure uniqueness
                while new_col in final_column_mapping.values():
                    seen[base_name] += 1
                    new_col = f"{base_name}_{seen[base_name]}"
                final_column_mapping[old_col] = new_col
            else:
                seen[base_name] = 0
                # Check if this name already exists in the final mapping values
                if base_name in final_column_mapping.values():
                    seen[base_name] = 1
                    new_col = f"{base_name}_{seen[base_name]}"
                    final_column_mapping[old_col] = new_col
                else:
                    final_column_mapping[old_col] = base_name
        
        # Rename columns in the dataframe
        empty_df = empty_df.rename(columns=final_column_mapping)
        
        print(f"Saving concatenated data to table {new_table_name}")
        
        # Instead of using to_sql directly, we'll use create_table then insert in chunks
        engine = create_long_running_engine(database)
        
        # First, create the table structure with one empty row to establish the schema
        if row_count > 0:
            # Use a small sample to create the table structure
            sample_df = empty_df.head(1)
            sample_df.to_sql(new_table_name, con=engine, if_exists='replace', index=False)
            
            # Drop the sample row if it was inserted
            cursor.execute(f"TRUNCATE TABLE `{new_table_name}`")
            connection.commit()
            
            # Now insert the data in chunks to avoid timeouts
            CHUNK_SIZE = 1000  # Adjust based on your table size and server capacity
            
            # Use raw SQL for faster inserts
            from sqlalchemy.dialects.mysql import insert
            from sqlalchemy import Table, MetaData, select
            from sqlalchemy.sql import text
            
            # Get the table metadata
            metadata = MetaData()
            metadata.reflect(bind=engine, only=[new_table_name])
            table = metadata.tables[new_table_name]
            
            # Insert data in chunks
            for start_idx in range(0, len(empty_df), CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, len(empty_df))
                chunk = empty_df.iloc[start_idx:end_idx]
                
                try:
                    # Convert DataFrame chunk to a list of dictionaries
                    records = chunk.to_dict('records')
                    
                    if records:
                        # Use the connection directly for better control
                        with engine.begin() as conn:
                            conn.execute(table.insert(), records)
                        
                    print(f"Inserted rows {start_idx} to {end_idx}")
                except Exception as e:
                    print(f"Error inserting chunk {start_idx}-{end_idx}: {e}")
                    # Try an alternative approach if the first method fails
                    try:
                        # Try with pandas to_sql for this chunk, with lower chunksize
                        chunk.to_sql(new_table_name, con=engine, if_exists='append', 
                                     index=False, chunksize=100)
                        print(f"Inserted rows {start_idx} to {end_idx} with alternative method")
                    except Exception as inner_e:
                        print(f"Alternative insert also failed: {inner_e}")
                        # Continue with next chunk instead of failing completely
        else:
            # Just create an empty table with the right structure
            empty_df.to_sql(new_table_name, con=engine, if_exists='replace', index=False)
        
        # Close the connection
        cursor.close()
        connection.close()

        return jsonify(success=True, message=f"Successfully concatenated {len(table_names)} tables with {row_count} rows each.")
    except Exception as e:
        print(f'Error in concatenate_tables: {e}')
        traceback.print_exc()  # Print the full traceback for debugging
        return jsonify(success=False, message=f"Error: {str(e)}")

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

        # Process each selected schema
        for schema in selected_schemas:
            # Get all tables from current schema
            cursor.execute(f"SHOW TABLES FROM `{schema}`")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                # Create new table name with schema prefix
                new_table_name = f"{schema}_{table_name}"
                
                # Check if the new table name exceeds MySQL's limit
                if len(new_table_name) > MAX_TABLE_NAME_LENGTH:
                    skipped_tables.append(f"{schema}.{table_name}")
                    continue
                
                # Check if table already exists in target schema
                cursor.execute(f"SHOW TABLES FROM `{new_schema_name}` LIKE %s", (new_table_name,))
                if cursor.fetchone():
                    duplicate_tables.append(f"{schema}.{table_name}")
                    continue
                
                # Copy table structure and data to new schema
                cursor.execute(f"""
                    CREATE TABLE `{new_schema_name}`.`{new_table_name}` 
                    LIKE `{schema}`.`{table_name}`
                """)
                cursor.execute(f"""
                    INSERT INTO `{new_schema_name}`.`{new_table_name}`
                    SELECT * FROM `{schema}`.`{table_name}`
                """)
                processed_tables.append(f"{schema}.{table_name}")

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
            message_parts.append(f'{len(duplicate_tables)} tables were skipped because they already exist in the target folder.')
            result['duplicate_tables'] = duplicate_tables
            
        result['message'] = ' '.join(message_parts)
        return jsonify(result)
        
    except Exception as e:
        print(f'Error merging schemas: {e}')
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
            'color_map_flag', 'outlier_analysis_flag', 'target_values', 'custom_division'  # Added custom_division here
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
    form_data['outlier_analysis_flag'] = form_data.get('outlier_analysis_flag', 'False') == 'True'
    form_data['custom_division'] = form_data.get('custom_division', 'False') == 'True'

    # Process target values
    target_values_str = form_data.get('target_values', '')
    if target_values_str:
        try:
            form_data['target_values'] = [float(x.strip()) for x in target_values_str.split(',') if x.strip()]
        except ValueError:
            form_data['target_values'] = []
    else:
        form_data['target_values'] = []

    print("Final Form Data:", form_data)  # Debug print
    return form_data

def get_form_data_generate_plot_ber_by_bls(form):
    return form_data

def get_form_data_generate_plot_read_stability(form):
    # Initialize form_data dictionary
    form_data = {}

    # Retrieve and store the 'state_pattern' from the form
    form_data['state_pattern'] = form.get('state_pattern', None)
    form_data['input_integer'] = form.get('input_integer', None)

    # Debug print to check the retrieved 'state_pattern'
    print("State Pattern:", form_data['state_pattern'])
    print("input_integer:", form_data['input_integer'])

    return form_data

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
        mysql_path = "/var/lib/mysql"  # Default MySQL data directory on Linux
        
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
                    
                    # Process exists - update URL to use correct hostname instead of localhost
                    hostname = socket.gethostname()
                    
                    # Process exists
                    return jsonify({
                        "status": "running",
                        "url": f"http://{hostname}:8888",
                        "token": status.get('token', '')
                    })
                except Exception as e:
                    # Process doesn't exist
                    return jsonify({
                        "status": "stopped",
                        "message": f"Process not running. Please restart the systemd service: sudo systemctl restart jupyter_notebook.service"
                    })
        except Exception as e:
            # Error reading status file
            return jsonify({
                "status": "error",
                "message": f"Error checking status: {str(e)}. Please restart the systemd service."
            })
    
    # No status file
    return jsonify({
        "status": "not_started",
        "message": "Jupyter server has not been started. Please run the systemd service."
    })

@app.route('/test-machines')
def test_machines():
    """
    Page to display the available test machines
    """
    test_machines = [
        {'ip': '192.168.68.124', 'user': 'slate', 'hostname': 'ARM Tester'},
        {'ip': '192.168.68.234', 'user': 'tc4', 'hostname': 'TC4'},
        {'ip': '192.168.68.129', 'user': 'nuc14', 'hostname': 'NUC14'},
        {'ip': '192.168.68.206', 'user': 'nuc6', 'hostname': 'NUC6'},
        {'ip': '192.168.68.164', 'user': 'lenovoi7', 'hostname': 'lenovoi7'},
        {'ip': '192.168.68.205', 'user': 'nuc5', 'hostname': 'NUC5'}
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
                '192.168.68.205': '2222'
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
        
        # Get total database size - it returns (total_size_bytes, formatted_size)
        try:
            _, total_db_size = get_total_database_size()
        except Exception as e:
            print(f"Error getting database size: {e}")
            total_db_size = "Unknown"
            
        return render_template('database_stats.html', 
                              databases=databases,
                              total_size=total_db_size,
                              available_space=available_space,
                              disk_info=disk_info)
                              
    except Exception as e:
        print(f"Error in database_stats: {e}")
        return f"Error loading database statistics: {str(e)}"