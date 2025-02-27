import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from scipy.interpolate import interp1d
import base64
from io import BytesIO
import pandas as pd
import os

def sigma_to_ppm(sigma):
    # Calculate the area in the tail beyond the sigma value on one side of the distribution
    tail_probability = sp_stats.norm.sf(sigma)
    # Convert this probability to parts per million
    ppm = tail_probability * 1_000_000
    return ppm
   
def get_group_data_new(selected_groups, file_name, sub_array_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    # Load data from file
    if file_path.endswith('.npy'):
        try:
            data = np.load(file_path)
        except Exception as e:
            raise IOError(f"Error loading .npy file: {e}")
    elif file_path.endswith('.csv'):
        try:
            # Load CSV data and exclude the first row (header)
            data = pd.read_csv(file_path, header=None).values
            data = data[1:]  # Exclude the first row
        except Exception as e:
            raise IOError(f"Error loading .csv file: {e}")
    else:
        raise ValueError("Unsupported file format. Please provide a .npy or .csv file.")
   
    # Convert fetched data to a NumPy array for easier manipulation
    data_np = np.array(data)

    # Accessing the first and second dimensions
    first_dimension = data_np.shape[0]
    second_dimension = data_np.shape[1]

    print("First dimension:", first_dimension)
    print("Second dimension:", second_dimension)

    # Handle zero values
    data_np[data_np == 0] = 0.001

    groups = []
    groups_stats = []  # List to store statistics for each group

    rows_per_group, cols_per_group = sub_array_size
    total_rows, total_cols = data_np.shape
    print("total_rows:", total_rows)
    print("total_cols:", total_cols)

    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    num_of_groups = num_col_groups * num_row_groups
    partial_rows = total_rows % rows_per_group  # Check if there's a partial row group
    partial_cols = total_cols % cols_per_group  # Check if there's a partial column group
    print("Number of row groups:", num_row_groups)
    print("Number of column groups:", num_col_groups)
    print("Total number of groups:", num_of_groups)
    print("Number of partial rows:", partial_rows)
    print("Number of partial columns:", partial_cols)

    group_idx = 0  # Initialize group index
    real_selected_groups = []

    for i in range(num_row_groups + (1 if partial_rows > 0 else 0)):
        for j in range(num_col_groups + (1 if partial_cols > 0 else 0)):
            start_row = i * rows_per_group
            end_row = (i + 1) * rows_per_group if i < num_row_groups else total_rows

            start_col = j * cols_per_group
            end_col = (j + 1) * cols_per_group if j < num_col_groups else total_cols

            # Check if this group is selected
            if group_idx in selected_groups:
                real_selected_groups.append(group_idx)

                try:
                    group = data_np[start_row:end_row, start_col:end_col]
                    flattened_group = group.flatten()

                    # Filter out negative values
                    positive_flattened_group = flattened_group[flattened_group >= 0]

                    groups.append(positive_flattened_group)

                    # Calculate statistics for the positive values
                    if len(positive_flattened_group) > 0:  # Ensure there are positive values to analyze
                        average = round(np.mean(positive_flattened_group), 2)
                        groups_stats.append((group_idx, average))
                    else:
                        print(f"State {group_idx} has no positive values for analysis.")
                except IndexError as e:
                    print(f"Error accessing data slice: {e}")

            group_idx += 1  # Increment group index after each inner loop

    def transform_list_by_order(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x])
        transformation = [0] * len(lst)
        for rank, index in enumerate(sorted_indices):
            transformation[index] = rank
        return transformation

    # Sort the groups and groups_stats based on the average value of the group
    groups_stats.sort(key=lambda x: x[1])  # Sort by average value
    sorted_indices = [i[0] for i in groups_stats]  # Get the sorted indices
    sorted_indices = transform_list_by_order(sorted_indices)
    groups = [groups[i] for i in sorted_indices]

    return groups
   
def plot_transformed_cdf_2(data, selected_groups):
    added_to_legend = set()
    ber_results = []
    transformed_data_groups = []
    global_x_min = float('inf')
    global_x_max = float('-inf')
    transformed_data = []

    for j, subgroup in enumerate(data):
        state_index = selected_groups[j]
        sorted_data = np.sort(subgroup)
        global_x_min = min(global_x_min, sorted_data[0])  # Update global x-axis minimum
        global_x_max = max(global_x_max, sorted_data[-1])  # Update global x-axis maximum

        cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        sigma_values = sp_stats.norm.ppf(cdf_values)

        transformed_data.append((sorted_data, sigma_values))

    intersections = []
    horizontal_line_y_value = []

    for k in range(len(transformed_data) - 1):
        x1, y1 = transformed_data[k]
        x2, y2 = transformed_data[k + 1]

        y1 = -y1  # Reverse the y-axis for the first of the two states being compared

        start_state = selected_groups[k]
        end_state = selected_groups[k + 1]

        common_x_min_all = min(min(x1), min(x2))
        common_x_max_all = max(max(x1), max(x2))
        common_x_all = np.linspace(common_x_min_all, common_x_max_all, num=5000)

        # Remove duplicates and interpolate
        unique_x1, unique_indices_x1 = np.unique(x1, return_index=True)
        unique_y1 = y1[unique_indices_x1]
        unique_x2, unique_indices_x2 = np.unique(x2, return_index=True)
        unique_y2 = y2[unique_indices_x2]

        interp_common_x_1 = interp1d(unique_x1, unique_y1, fill_value="extrapolate")(common_x_all)
        interp_common_x_2 = interp1d(unique_x2, unique_y2, fill_value="extrapolate")(common_x_all)

        # Don't Convert sigma to CDF values, keep sigma values
        cdf_value_1 = interp_common_x_1
        cdf_value_2 = interp_common_x_2

        # Check if both cdf_value_1 and cdf_value_2 are not all NaN before plotting and finding intersections
        if not (np.isnan(cdf_value_1).all() or np.isnan(cdf_value_2).all()):
            # Find and mark intersection only if both arrays have valid data
            idx_closest = np.argmin(np.abs(cdf_value_1 - cdf_value_2))
            intersection_x = common_x_all[idx_closest]
            intersection_y = cdf_value_1[idx_closest]
            plt.scatter(intersection_x, intersection_y, color='red', s=50, zorder=5)
            intersections.append((intersection_x, intersection_y))

            ber = np.abs(cdf_value_1[idx_closest])
            ppm_ber = sigma_to_ppm(ber)  #intersection

            # Draw horizontal lines if x-differences are about 2 units apart
            target_x_diff = 2
            tolerance = 0.2
            line_drawn = False

            for idx in range(len(common_x_all) - 1):
                for jdx in range(idx + 1, len(common_x_all)):
                    x_diff = common_x_all[jdx] - common_x_all[idx]
                    if abs(x_diff - target_x_diff) < tolerance:
                        if cdf_value_2[jdx] > cdf_value_1[idx]:  # Check divergence
                            horizontal_line_y_value = cdf_value_2[jdx]
                            ppm = sigma_to_ppm(abs(horizontal_line_y_value))  #2uS
                            print(f"Horizontal line drawn from x={common_x_all[idx]} to x={common_x_all[jdx]} at y={cdf_value_2[jdx]}")
                            #print("ppm:", ppm)
                            line_drawn = True
                            break
	                    else:
	                        horizontal_line_y_value = None
	                        #ppm = sigma_to_ppm(abs(horizontal_line_y_value))
	                        ppm = None
                if line_drawn:
                    break

            if not line_drawn:
                print("No suitable points found to draw a horizontal line.")
        else:
            ber = 0
            ppm_ber = 0
            ppm = 0
            horizontal_line_y_value = 0

        if horizontal_line_y_value is not None:
            hlyv_rounded = round(abs(horizontal_line_y_value), 4)
        else:
            hlyv_rounded = None  # or 0, depending on your preference
        #ber_results.append(('', f'state{start_state} to state{end_state}', ber, ppm_ber, ppm, round(abs(horizontal_line_y_value), 4)))
        ber_results.append((f'state{start_state} to state{end_state}', ppm_ber, ppm))

    print("ber_results:", ber_results)
   
#selected_groups = [0, 2, 3]
selected_groups = [0, 1, 2, 3]
file_name = 'data.csv'
sub_array_size = (324, 64)
groups = get_group_data_new(selected_groups, file_name, sub_array_size)
print(groups)
plot_transformed_cdf_2(groups, selected_groups)