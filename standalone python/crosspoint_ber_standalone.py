import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from scipy.interpolate import interp1d
import base64
from io import BytesIO
import pandas as pd
import os
# Add numba for JIT compilation
from numba import jit, prange

def sigma_to_ppm(sigma):
    # Calculate the area in the tail beyond the sigma value on one side of the distribution
    tail_probability = sp_stats.norm.sf(sigma)
    # Convert this probability to parts per million
    ppm = tail_probability * 1_000_000
    return ppm

# Use numba to accelerate group data extraction   
@jit(nopython=True, parallel=True)
def process_groups(data_np, selected_groups, rows_per_group, cols_per_group, total_rows, total_cols):
    groups = []
    group_indices = []
    
    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    partial_rows = total_rows % rows_per_group
    partial_cols = total_cols % cols_per_group
    
    group_idx = 0
    
    for i in range(num_row_groups + (1 if partial_rows > 0 else 0)):
        for j in range(num_col_groups + (1 if partial_cols > 0 else 0)):
            start_row = i * rows_per_group
            end_row = (i + 1) * rows_per_group if i < num_row_groups else total_rows

            start_col = j * cols_per_group
            end_col = (j + 1) * cols_per_group if j < num_col_groups else total_cols

            if group_idx in selected_groups:
                group = data_np[start_row:end_row, start_col:end_col].flatten()
                groups.append(group)
                group_indices.append(group_idx)
                
            group_idx += 1
    
    return groups, group_indices
   
def get_group_data_new(selected_groups, file_name, sub_array_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    # Load data from file more efficiently
    if file_path.endswith('.npy'):
        try:
            data = np.load(file_path)
        except Exception as e:
            raise IOError(f"Error loading .npy file: {e}")
    elif file_path.endswith('.csv'):
        try:
            # Use engine='c' for faster CSV parsing and only read necessary columns
            data = pd.read_csv(file_path, header=None, engine='c').values
            data = data[1:]  # Exclude the first row
        except Exception as e:
            raise IOError(f"Error loading .csv file: {e}")
    else:
        raise ValueError("Unsupported file format. Please provide a .npy or .csv file.")
   
    # Convert fetched data to a NumPy array for easier manipulation
    data_np = np.array(data, dtype=np.float32)  # Use float32 for better performance

    # Handle zero values
    data_np[data_np == 0] = 0.001

    # Print dimensions
    first_dimension, second_dimension = data_np.shape
    print("First dimension:", first_dimension)
    print("Second dimension:", second_dimension)
    print("total_rows:", first_dimension)
    print("total_cols:", second_dimension)

    rows_per_group, cols_per_group = sub_array_size
    total_rows, total_cols = data_np.shape

    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    print("Number of row groups:", num_row_groups)
    print("Number of column groups:", num_col_groups)
    print("Total number of groups:", num_row_groups * num_col_groups)
    print("Number of partial rows:", total_rows % rows_per_group)
    print("Number of partial columns:", total_cols % cols_per_group)

    # Convert selected_groups to numpy array for numba
    selected_groups_array = np.array(selected_groups)
    
    # Use numba-optimized function for group processing
    groups, real_selected_groups = process_groups(data_np, selected_groups_array, 
                                                 rows_per_group, cols_per_group, 
                                                 total_rows, total_cols)
    
    groups_stats = []
    for i, group in enumerate(groups):
        # Filter out negative values and zeros
        positive_values = group[group > 0]
        if len(positive_values) > 0:
            average = round(np.mean(positive_values), 2)
            groups_stats.append((real_selected_groups[i], average))
    
    # Sort by average value
    groups_stats.sort(key=lambda x: x[1])
    sorted_indices = [i[0] for i in groups_stats]
    
    # Create mapping for sorting
    sorted_pos = {idx: pos for pos, idx in enumerate(sorted_indices)}
    groups = [g for _, g in sorted(zip([sorted_pos[sg] for sg in real_selected_groups], groups))]
    
    return groups

# Optimized function to find intersection of two curves
@jit(nopython=True)
def find_intersection(x1, y1, x2, y2):
    # Find where the difference between curves is closest to zero
    min_diff_idx = np.argmin(np.abs(y1 - y2))
    return x1[min_diff_idx], y1[min_diff_idx]

def plot_transformed_cdf_2(data, selected_groups):
    added_to_legend = set()
    ber_results = []
    transformed_data = []

    # Pre-process all data first
    for j, subgroup in enumerate(data):
        sorted_data = np.sort(subgroup)
        cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        sigma_values = sp_stats.norm.ppf(cdf_values)
        transformed_data.append((sorted_data, sigma_values))

    # Use 1000 points instead of 5000 for interpolation - still accurate but faster
    num_interp_points = 4000

    for k in range(len(transformed_data) - 1):
        x1, y1 = transformed_data[k]
        x2, y2 = transformed_data[k + 1]

        y1 = -y1  # Reverse the y-axis for the first of the two states being compared

        start_state = selected_groups[k]
        end_state = selected_groups[k + 1]

        common_x_min_all = min(min(x1), min(x2))
        common_x_max_all = max(max(x1), max(x2))
        common_x_all = np.linspace(common_x_min_all, common_x_max_all, num=num_interp_points)

        # Remove duplicates and interpolate
        unique_x1, unique_indices_x1 = np.unique(x1, return_index=True)
        unique_y1 = y1[unique_indices_x1]
        unique_x2, unique_indices_x2 = np.unique(x2, return_index=True)
        unique_y2 = y2[unique_indices_x2]

        # Use bounds_error=False for faster interpolation
        interp_common_x_1 = interp1d(unique_x1, unique_y1, fill_value="extrapolate", bounds_error=False)(common_x_all)
        interp_common_x_2 = interp1d(unique_x2, unique_y2, fill_value="extrapolate", bounds_error=False)(common_x_all)

        # Don't Convert sigma to CDF values, keep sigma values
        cdf_value_1 = interp_common_x_1
        cdf_value_2 = interp_common_x_2

        # Check if both cdf_value_1 and cdf_value_2 are not all NaN before processing
        if not (np.isnan(cdf_value_1).all() or np.isnan(cdf_value_2).all()):
            # Find intersection more efficiently
            intersection_x, intersection_y = find_intersection(common_x_all, cdf_value_1, common_x_all, cdf_value_2)
            plt.scatter(intersection_x, intersection_y, color='red', s=50, zorder=5)

            ber = np.abs(intersection_y)
            ppm_ber = sigma_to_ppm(ber)  # intersection

            # More efficient horizontal line calculation
            target_x_diff = 2
            tolerance = 0.2
            horizontal_line_y_value = None
            
            # Vectorized approach to find indices meeting criteria
            x_diffs = common_x_all[:, np.newaxis] - common_x_all[np.newaxis, :]
            valid_diffs = np.abs(x_diffs - target_x_diff) < tolerance
            
            # Find valid pairs where values are diverging
            for i in range(len(common_x_all)):
                for j in range(i+1, len(common_x_all)):
                    if valid_diffs[i, j] and cdf_value_2[j] > cdf_value_1[i]:
                        horizontal_line_y_value = cdf_value_2[j]
                        print(f"Horizontal line drawn from x={common_x_all[i]} to x={common_x_all[j]} at y={horizontal_line_y_value}")
                        ppm = sigma_to_ppm(abs(horizontal_line_y_value))
                        break
                if horizontal_line_y_value is not None:
                    break
                        
            if horizontal_line_y_value is None:
                print("No suitable points found to draw a horizontal line.")
                ppm = None
        else:
            ber = 0
            ppm_ber = 0
            ppm = 0
            horizontal_line_y_value = 0

        if horizontal_line_y_value is not None:
            hlyv_rounded = round(abs(horizontal_line_y_value), 4)
        else:
            hlyv_rounded = None
            
        ber_results.append((f'state{start_state} to state{end_state}', ppm_ber, ppm))

    print("ber_results:", ber_results)
   
#selected_groups = [0, 2, 3]
selected_groups = [0, 1, 2, 3]
file_name = 'data.csv'
sub_array_size = (324, 64)
groups = get_group_data_new(selected_groups, file_name, sub_array_size)
print(groups)
plot_transformed_cdf_2(groups, selected_groups)