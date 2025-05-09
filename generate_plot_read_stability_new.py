from db_operations import DB_CONFIG
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import io
import base64
import os
from io import BytesIO
from tools_for_plots import *
from conductance_calculator import convert_table_to_conductance, convert_table_to_linear

def generate_plot_read_stability(table_names, database_name, form_data):
    # Extract form data
    state_pattern = form_data.get('state_pattern')
    input_integer = form_data.get('input_integer', '100')  # Default to 100 if not provided
    using_conductance = form_data.get('using_conductance', False)
    using_linear_conversion = form_data.get('using_linear_conversion', False)
    conductance_params = form_data.get('conductance_params', {})
    
    print("table_names:", table_names)
    print("state_pattern:", state_pattern)
    print("input_integer:", input_integer)
    print("using_conductance:", using_conductance)
    print("using_linear_conversion:", using_linear_conversion)
    
    # Define the path to your state pattern files directory
    pattern_files = {
        "1296x64_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_rowbar_4states.npy",
        "3x4_4states_debug": "/home/admin2/webapp_2/State_pattern_files/3x4_4states_debug.npy",
        "248x248_checkerboard_4states": "/home/admin2/webapp_2/State_pattern_files/248x248_checkerboard_4states.npy",
        "1296x64_Adrien_random_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_Adrien_random_4states.npy",
        "248x248_1state": "/home/admin2/webapp_2/State_pattern_files/248x248_1state.npy",
        "1296x64_1state": "/home/admin2/webapp_2/State_pattern_files/1296x64_1state.npy",
        "248x248_16states": "/home/admin2/webapp_2/State_pattern_files/248x248_16states.npy",
        "248x1_1state": "/home/admin2/webapp_2/State_pattern_files/248x1_1state.npy",
        "82944x78_ecc_fuxi": "/home/admin2/webapp_2/State_pattern_files/82944x78_ecc_fuxi.npy"
    }

    # Fetch the file path based on the state pattern
    file_path = pattern_files.get(state_pattern)

    # Load the pattern file array if the file path is found
    if file_path:
        pattern_file_array = np.load(file_path)
        # Special handling for 82944x78_ecc_fuxi.npy which is actually (78, 1296, 64)
        if state_pattern == "82944x78_ecc_fuxi":
            # Reshape the 3D array to 2D (78, 82944) and then transpose to (82944, 78)
            pattern_file_array = pattern_file_array.reshape(78, 82944).T
    else:
        print("Invalid state pattern or file path not found.")
        return []

    # Initialize an empty list to hold the encoded plots
    encoded_plots = []
    global_min_max_values = []

    for table_name in table_names:
        # Get the data from the table
        data_matrix, _ = get_full_table_data(table_name, database_name)
        
        # Apply conductance conversion if enabled
        if using_conductance and conductance_params:
            print(f"Converting table {table_name} to conductance values")
            data_matrix = convert_table_to_conductance(data_matrix, conductance_params)
        # Apply linear conversion if enabled
        elif using_linear_conversion:
            print(f"Converting table {table_name} using linear conversion (0-63 → 60-170)")
            data_matrix = convert_table_to_linear(data_matrix)
            
        # Store min and max values for the global range
        min_val = np.min(data_matrix)
        max_val = np.max(data_matrix)
        global_min_max_values.append((min_val, max_val))

    # Calculate global min and max
    global_min = min(val[0] for val in global_min_max_values)
    global_max = max(val[1] for val in global_min_max_values)
    g_range = (global_min, global_max)

    for table_name in table_names:
        # Get the data from the table
        data_matrix, _ = get_full_table_data(table_name, database_name)
        
        # Apply conductance conversion if enabled
        if using_conductance and conductance_params:
            print(f"Converting table {table_name} to conductance values")
            data_matrix = convert_table_to_conductance(data_matrix, conductance_params)
        # Apply linear conversion if enabled
        elif using_linear_conversion:
            print(f"Converting table {table_name} using linear conversion (0-63 → 60-170)")
            data_matrix = convert_table_to_linear(data_matrix)

        # Create separate matrices for each state
        unique_states = np.unique(pattern_file_array)
        state_matrices = {}

        for state in unique_states:
            # Create a mask for the current state
            mask = (pattern_file_array == state)
            
            # Create a matrix for this state with NaN values where the mask is False
            state_matrix = np.full_like(data_matrix, np.nan, dtype=float)
            state_matrix[mask] = data_matrix[mask]
            
            # Store the state matrix
            state_matrices[int(state)] = state_matrix

        # Plot read stability for this table
        plots = plot_read_stability(table_name, state_matrices, int(input_integer), g_range)
        encoded_plots.extend(plots)

    return encoded_plots

def plot_read_stability(table_name, state_matrices, max_y_value, g_range):
    encoded_plots = []

    # Create a colormap plot for each state
    for state, matrix in state_matrices.items():
        # Skip matrices that are all NaN
        if np.all(np.isnan(matrix)):
            continue
            
        # Create a plot for this state
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use imshow to create a heatmap with the global range
        im = ax.imshow(matrix, cmap='viridis', vmin=g_range[0], vmax=g_range[1])
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        # Set the title
        ax.set_title(f"{table_name} - State {state}")
        
        # Add axis labels
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the buffer as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        encoded_plots.append(f"data:image/png;base64,{img_str}")
        
        # Clear the figure to avoid memory leaks
        plt.close(fig)

    # Create histograms for each state
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (state, matrix) in enumerate(state_matrices.items()):
        # Skip matrices that are all NaN
        if np.all(np.isnan(matrix)):
            continue
            
        # Flatten the matrix and remove NaN values
        values = matrix.flatten()
        values = values[~np.isnan(values)]
        
        # Create a histogram
        color = colors[i % len(colors)]
        ax.hist(values, bins=50, alpha=0.5, label=f'State {state}', color=color)
    
    # Set the title and labels
    ax.set_title(f'Histogram of Values by State - {table_name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Set y-axis limit if specified
    if max_y_value and int(max_y_value) > 0:
        ax.set_ylim(0, int(max_y_value))
    
    # Add a legend
    ax.legend()
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the buffer as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    encoded_plots.append(f"data:image/png;base64,{img_str}")
    
    # Clear the figure to avoid memory leaks
    plt.close(fig)

    return encoded_plots 