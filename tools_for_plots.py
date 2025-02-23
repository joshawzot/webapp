# Standard library imports
from datetime import datetime
from io import BytesIO
import base64
import re

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy.stats as sp_stats  # Alias for scipy.stats
from scipy.stats import gamma
from scipy.integrate import quad
from fitter import Fitter
from PIL import Image  # For image processing

# Local application imports
from db_operations import create_connection, close_connection

def plot_boxplot(data, table_names, figsize=(15, 10)):
    # Create a new figure instance for this plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    try:
        # Calculate positions for each box
        xticks = []
        xticklabels = table_names
        
        # Iterate over each group to plot
        for i, group in enumerate(data):
            # Calculate the first position of each group for the x-tick
            start_position = i * len(group) + 1
            xticks.append(start_position)

            # Plot each box in the group
            for j, subgroup in enumerate(group):
                position = i * len(group) + j + 1
                ax.boxplot(subgroup, positions=[position], widths=0.6)

        # Set x-axis ticks and labels
        if len(xticks) == len(xticklabels):
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, fontsize=12)
        else:
            print("Error: Mismatch in the number of xticks and xticklabels.")

        # Configure plot appearance
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True)

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_histogram(data, table_names, colors, figsize=(15, 10)):
    # Create a new figure instance for this plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    try:
        # Calculate global min and max for consistent binning
        global_min = min([min(subgroup) for group in data for subgroup in group])
        global_max = max([max(subgroup) for group in data for subgroup in group])

        # Create bin edges with an increment of 1
        bin_edges = np.arange(global_min, global_max + 1, 1)

        # Track filenames added to legend
        added_to_legend = set()

        # Plot histograms for each group
        for i, group in enumerate(data):
            for j, subgroup in enumerate(group):
                # Only add label the first time a filename is encountered
                label = f'{table_names[i]}' if table_names[i] not in added_to_legend else None
                if label:
                    added_to_legend.add(table_names[i])

                # Generate histogram for each subgroup
                ax.hist(subgroup, bins=bin_edges, color=colors[i], alpha=0.75, 
                       label=label, log=True)

        # Configure plot appearance
        ax.set_ylabel('Frequency', fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

'''
I want to also get the BER between different states for each table_names, for example for a specific table_name,
if it has 4 states of data, state0, state1, state2, state3, there will be 3 BER, one between state0 and state1, between state1 and state2, between state2 and state3.
To calculate the BER,
reverse the y-axis (for example 1 becomes -1, -2 becomes 2) of state0 and record the absolute value of y-axis that the transformed cdf plot of state0 and the transformed cdf plot of state1 intersects.
reverse the y-axis (for example 1 becomes -1, -2 becomes 2) of state1 and record the absolute value of y-axis that the transformed cdf plot of state1 and the transformed cdf plot of state2 intersects.
reverse the y-axis (for example 1 becomes -1, -2 becomes 2) of state2 and record the absolute value of y-axis that the transformed cdf plot of state2 and the transformed cdf plot of state3 intersects.
the absolute value of y-axis of intersects are the BER.
'''

def sigma_to_ppm(sigma):
    # Calculate the area in the tail beyond the sigma value on one side of the distribution
    tail_probability = sp_stats.norm.sf(sigma)
    # Convert this probability to parts per million
    ppm = tail_probability * 1_000_000
    return ppm

from scipy.stats import norm 
def cdf_to_ppm(cdf): 
    tail_probability = cdf
    ppm = tail_probability * 1_000_000 
    return ppm

from scipy.stats import norm 
def sigma_to_cdf(sigma): 
    # Calculate the CDF for the given sigma value 
    cdf_value = norm.cdf(sigma) 
    return cdf_value

from scipy.interpolate import interp1d
def calculate_sigma_intersections(sorted_data, sigma_values, target_sigmas=[-4, -3, -2, -1, 0, 1, 2, 3, 4]):
    """Calculate x values where the distribution intersects with integer sigma values."""
    # Create interpolation function
    interp_func = interp1d(sigma_values, sorted_data, bounds_error=False, fill_value=np.nan)
    
    # Find x values at target sigma points
    x_at_sigmas = interp_func(target_sigmas)
    
    return x_at_sigmas

def plot_transformed_cdf_2(data, table_names, selected_groups, colors, figsize=(15, 10)):
    # Initialize variables
    added_to_legend = set()
    ber_results = []
    transformed_data_groups = []
    global_x_min = float('inf')
    global_x_max = float('-inf')
    sigma_intersections = {}  # Store sigma intersections for each table and state

    # Create colormap
    num_colors = max(len(data), 20)
    colormap = plt.get_cmap('tab20', num_colors)
    color_normalizer = mcolors.Normalize(vmin=0, vmax=num_colors - 1)
    scalar_map = plt.cm.ScalarMappable(norm=color_normalizer, cmap=colormap)

    # Create separate figures for sigma and CDF plots
    fig_sigma = plt.figure(figsize=figsize)
    ax_sigma = fig_sigma.add_subplot(111)
    
    fig_cdf = plt.figure(figsize=figsize)
    ax_cdf = fig_cdf.add_subplot(111)
    
    try:
        # Process data and create transformed plots
        for i, group in enumerate(data):
            transformed_data = []
            color = scalar_map.to_rgba(i)
            table_name = table_names[i]
            sigma_intersections[table_name] = []

            for j, subgroup in enumerate(group):
                state_index = selected_groups[j]
                label = table_name if table_name not in added_to_legend else None
                if label:
                    added_to_legend.add(label)

                sorted_data = np.sort(subgroup)
                global_x_min = min(global_x_min, sorted_data[0])
                global_x_max = max(global_x_max, sorted_data[-1])

                cdf_values = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
                sigma_values = sp_stats.norm.ppf(cdf_values)

                # Calculate sigma intersections for this state
                x_at_sigmas = calculate_sigma_intersections(sorted_data, sigma_values)
                sigma_intersections[table_name].append(x_at_sigmas)

                # Plot sigma values
                ax_sigma.plot(sorted_data, sigma_values, linestyle='-', linewidth=1, color=color, label=label)
                ax_sigma.scatter(sorted_data, sigma_values, s=10, color=color)

                # Plot CDF values
                ax_cdf.plot(sorted_data, cdf_values, linestyle='-', linewidth=1, color=color, label=label)
                ax_cdf.scatter(sorted_data, cdf_values, s=10, color=color)

                transformed_data.append((sorted_data, sigma_values))

            transformed_data_groups.append(transformed_data)

        # Configure sigma plot
        ax_sigma.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
        ax_sigma.grid(True)
        ax_sigma.set_ylabel('Sigma')
        ax_sigma.set_xlabel('Value')

        # Configure CDF plot
        ax_cdf.set_yscale('log')
        ax_cdf.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
        ax_cdf.grid(True)

        # Save sigma plot
        buf_sigma = BytesIO()
        fig_sigma.savefig(buf_sigma, format='png', bbox_inches='tight')
        buf_sigma.seek(0)
        plot_data_sigma = base64.b64encode(buf_sigma.getvalue()).decode('utf-8')

        # Save CDF plot
        buf_cdf = BytesIO()
        fig_cdf.savefig(buf_cdf, format='png', bbox_inches='tight')
        buf_cdf.seek(0)
        plot_data_cdf = base64.b64encode(buf_cdf.getvalue()).decode('utf-8')

        # Create interpolated CDF plot
        fig_interp = plt.figure(figsize=figsize)
        ax_interp = fig_interp.add_subplot(111)
        
        try:
            ax_interp.set_xlim(global_x_min, global_x_max)

            intersections = []
            horizontal_line_y_value = []

            for i, transformed_data in enumerate(transformed_data_groups):
                color = scalar_map.to_rgba(i)

                for k in range(len(transformed_data) - 1):
                    x1, y1 = transformed_data[k]
                    x2, y2 = transformed_data[k + 1]

                    y1 = -y1  # Reverse y-axis for first state

                    start_state = selected_groups[k]
                    end_state = selected_groups[k + 1]
                    table_name = table_names[i]

                    common_x_min_all = min(min(x1), min(x2))
                    common_x_max_all = max(max(x1), max(x2))
                    common_x_all = np.linspace(common_x_min_all, common_x_max_all, num=4000)

                    # Remove duplicates and interpolate
                    unique_x1, unique_indices_x1 = np.unique(x1, return_index=True)
                    unique_y1 = y1[unique_indices_x1]
                    unique_x2, unique_indices_x2 = np.unique(x2, return_index=True)
                    unique_y2 = y2[unique_indices_x2]

                    interp_common_x_1 = interp1d(unique_x1, unique_y1, fill_value="extrapolate")(common_x_all)
                    interp_common_x_2 = interp1d(unique_x2, unique_y2, fill_value="extrapolate")(common_x_all)

                    cdf_value_1 = interp_common_x_1
                    cdf_value_2 = interp_common_x_2

                    if not (np.isnan(cdf_value_1).all() or np.isnan(cdf_value_2).all()):
                        ax_interp.plot(common_x_all, cdf_value_1, linestyle='-', color=color, alpha=0.7, 
                                     label=f'{table_name} - state {start_state}')
                        ax_interp.plot(common_x_all, cdf_value_2, linestyle='-', color=color, alpha=0.7, 
                                     label=f'{table_name} - state {end_state}')

                        # Find and mark intersection
                        idx_closest = np.argmin(np.abs(cdf_value_1 - cdf_value_2))
                        intersection_x = common_x_all[idx_closest]
                        intersection_y = cdf_value_1[idx_closest]
                        ax_interp.scatter(intersection_x, intersection_y, color='red', s=50, zorder=5)
                        intersections.append((intersection_x, intersection_y))

                        ber = np.abs(cdf_value_1[idx_closest])
                        ppm_ber = sigma_to_ppm(ber)

                        # Draw horizontal lines for divergence analysis
                        target_x_diff = 2
                        tolerance = 0.2
                        line_drawn = False

                        for idx in range(len(common_x_all) - 1):
                            for jdx in range(idx + 1, len(common_x_all)):
                                x_diff = common_x_all[jdx] - common_x_all[idx]
                                if abs(x_diff - target_x_diff) < tolerance:
                                    if cdf_value_2[jdx] > cdf_value_1[idx]:
                                        ax_interp.hlines(y=cdf_value_2[jdx], xmin=common_x_all[idx], 
                                                       xmax=common_x_all[jdx], color='green', linestyles='dotted')
                                        horizontal_line_y_value = cdf_value_2[jdx]
                                        ppm = sigma_to_ppm(abs(horizontal_line_y_value))
                                        line_drawn = True
                                        break
                                    else:
                                        horizontal_line_y_value = None
                                        ppm = None

                            if line_drawn:
                                break

                        if not line_drawn:
                            horizontal_line_y_value = None
                            ppm = None
                    else:
                        ber = 0
                        ppm_ber = 0
                        ppm = 0
                        horizontal_line_y_value = 0

                    hlyv_rounded = round(abs(horizontal_line_y_value), 4) if horizontal_line_y_value is not None else None
                    ber_results.append((table_name, f'state{start_state} to state{end_state}', 
                                      ber, ppm_ber, ppm, hlyv_rounded, 4))

            ax_interp.grid(True)
            ax_interp.set_ylim(bottom=-8, top=8)
            ax_interp.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

            # Save interpolated CDF plot
            buf_interp = BytesIO()
            fig_interp.savefig(buf_interp, format='png', bbox_inches='tight')
            buf_interp.seek(0)
            plot_data_interpolated_cdf = base64.b64encode(buf_interp.getvalue()).decode('utf-8')

            return plot_data_sigma, plot_data_cdf, plot_data_interpolated_cdf, ber_results, sigma_intersections

        finally:
            plt.close(fig_interp)
            if 'buf_interp' in locals():
                buf_interp.close()

    finally:
        # Clean up resources
        plt.close(fig_sigma)
        plt.close(fig_cdf)
        if 'buf_sigma' in locals():
            buf_sigma.close()
        if 'buf_cdf' in locals():
            buf_cdf.close()

from db_operations import create_connection, fetch_data, close_connection, create_db_engine, create_db, get_all_databases, connect_to_db, fetch_tables, rename_database
def get_group_data_new(table_name, selected_groups, database_name, number_of_states):
    connection = create_connection(database_name)
    query = f"SELECT * FROM `{table_name}`"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
        
    # Convert fetched data to a NumPy array for easier manipulation
    data_np = np.array(data)
    data_np = data_np.astype(float)

    # Accessing the first and second dimensions
    first_dimension = data_np.shape[0] #82944
    second_dimension = data_np.shape[1] #2

    print("First dimension:", first_dimension)
    print("Second dimension:", second_dimension)

    data_np[data_np == 0] = 0.001

    # if np.mean(data_np) < 1:
    #     data_np = data_np * 1e6

    groups = []
    groups_stats = []  # List to store statistics for each group

    rows_per_group, cols_per_group = int(first_dimension/number_of_states), second_dimension
    total_rows, total_cols = data_np.shape

    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    num_of_groups = num_col_groups * num_row_groups
    partial_rows = total_rows % rows_per_group  # Check if there's a partial row group
    partial_cols = total_cols % cols_per_group  # Check if there's a partial column group

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
                        std_dev = round(np.std(positive_flattened_group), 2)
                        outlier_percentage = round(np.sum(np.abs(positive_flattened_group - average) > 2.698 * std_dev) / len(positive_flattened_group) * 100, 2)
                        groups_stats.append((table_name, group_idx, average, std_dev, outlier_percentage))
                    else:
                        print(f"State {group_idx} has no positive values for analysis.")
                except IndexError as e:
                    print(f"Error accessing data slice: {e}")

            group_idx += 1  # Increment group index after each inner loop

    close_connection()

    def transform_list_by_order(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x])
        transformation = [0] * len(lst)
        for rank, index in enumerate(sorted_indices):
            transformation[index] = rank
        return transformation

    # Sort the groups, groups_stats, and real_selected_groups based on the average value of the group
    groups_stats.sort(key=lambda x: x[2])  # Sort by average value
    sorted_indices = [i[1] for i in groups_stats]  #Get the sorted indices
    print("sorted_indices:", sorted_indices)
    sorted_indices = transform_list_by_order(sorted_indices)
    groups = [groups[i] for i in sorted_indices]
    #real_selected_groups = [real_selected_groups[i] for i in sorted_indices]
    #print('real_selected_groups:', real_selected_groups)

    return groups, groups_stats, real_selected_groups

def get_group_data_latest(target_ranges, table_name, selected_groups, database_name, number_of_states):
    connection = create_connection(database_name)
    query = f"SELECT * FROM {table_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()

    data_np = np.array(data)

    # Accessing the first and second dimensions
    first_dimension = data_np.shape[0] #82944
    second_dimension = data_np.shape[1] #2

    print("First dimension:", first_dimension)
    print("Second dimension:", second_dimension)

    data_np[data_np == 0] = 0.001  # Replace zeros with a small value to avoid issues

    # if np.mean(data_np) < 1:
    #     data_np = data_np * 1e6

    groups = []
    groups_2 = []
    groups_stats = []  # List to store statistics for each group

    rows_per_group, cols_per_group = int(first_dimension/number_of_states), second_dimension
    total_rows, total_cols = data_np.shape

    num_row_groups = total_rows // rows_per_group
    num_col_groups = total_cols // cols_per_group
    num_of_groups = num_col_groups * num_row_groups
    partial_rows = total_rows % rows_per_group  # Check if there's a partial row group
    partial_cols = total_cols % cols_per_group  # Check if there's a partial column group

    group_idx = 0  # Initialize group index
    real_selected_groups = []
    count = 0

    for i in range(num_row_groups + (1 if partial_rows > 0 else 0)):
        for j in range(num_col_groups + (1 if partial_cols > 0 else 0)):
            start_row = i * rows_per_group
            end_row = (i + 1) * rows_per_group if i < num_row_groups else total_rows

            start_col = j * cols_per_group
            end_col = (j + 1) * cols_per_group if j < num_col_groups else total_cols

            # Check if this group is selected
            #print("selected_groups:", selected_groups)
            #print("group_idx:", group_idx)
            if group_idx in selected_groups:
                real_selected_groups.append(group_idx)
                #print("real_selected_groups:", real_selected_groups)

                try:
                    print("start_row:", start_row)
                    print("end_row:", end_row)
                    print("start_col:", start_col)
                    print("end_col:", end_col)                    
                    group = data_np[start_row:end_row, start_col:end_col]
                    flattened_group = group.flatten()

                    # Filter out negative values
                    positive_flattened_group = flattened_group[flattened_group >= 0]
                    groups.append(positive_flattened_group)
                    groups_2.append(group)

                    # Calculate statistics for the positive values
                    if len(group) > 0:  # Ensure there are positive values to analyze
                        average = round(np.mean(group), 2)
                        std_dev = round(np.std(group), 2)

                        # Get the target range for this group
                        print("group_idx:", group_idx)
                        print("count:", count)
                        #lower_bound, upper_bound = target_ranges[group_idx * 2], target_ranges[group_idx * 2 + 1]
                        lower_bound, upper_bound = target_ranges[count * 2], target_ranges[count * 2 + 1]
                        print("lower_bound:", lower_bound)
                        print("upper_bound:", upper_bound)

                        # Calculate the BER (ppm value of data outside the target range)
                        out_of_range_data = group[
                            (group < lower_bound) | (group > upper_bound)
                        ]
                        ber_value = round(len(out_of_range_data) / len(group) * 1e6)  # Calculate ppm

                        # Store statistics including BER value
                        outlier_percentage = round(
                            np.sum(np.abs(group - average) > 2.698 * std_dev) / len(group) * 100, 0
                        )

                        groups_stats.append((table_name, group_idx, average, std_dev, outlier_percentage, ber_value))
                        count += 1
                    else:
                        print(f"State {group_idx} has no positive values for analysis.")
                except IndexError as e:
                    print(f"Error accessing data slice: {e}")

            group_idx += 1  # Increment group index after each inner loop

    close_connection()

    def transform_list_by_order(lst):
        sorted_indices = sorted(range(len(lst)), key=lambda x: lst[x])
        transformation = [0] * len(lst)
        for rank, index in enumerate(sorted_indices):
            transformation[index] = rank
        return transformation

    # Sort the groups, groups_stats, and real_selected_groups based on the average value of the group
    groups_stats.sort(key=lambda x: x[2])  # Sort by average value
    sorted_indices = [i[1] for i in groups_stats]  # Get the sorted indices
    print("sorted_indices:", sorted_indices)
    sorted_indices = transform_list_by_order(sorted_indices)
    groups = [groups[i] for i in sorted_indices]
    #print(groups)

   # Calculate BER values for different levels and transitions
    num_levels = len(selected_groups)
    ber_values = {}

    # Level n BER
    for level in range(num_levels):
        lower_bound, upper_bound = target_ranges[level * 2], target_ranges[level * 2 + 1]
        level_data = groups[level]
        if len(level_data) > 0:
            out_of_range_data = level_data[
                (level_data < lower_bound) | (level_data > upper_bound)
            ]
            ber_values[f"State{level}"] = round(len(out_of_range_data) / len(level_data) * 1e6)
        else:
            ber_values[f"State{level}"] = 0

    # Level n to level n-1 BER
    for level in range(1, num_levels):
        lower_bound = target_ranges[level * 2]
        level_data = groups[level]
        if len(level_data) > 0:
            ber_values[f"State{level} to State{level-1}"] = round(
                np.sum(level_data < lower_bound) / len(level_data) * 1e6
            )
        else:
            ber_values[f"State{level} to Level{level-1}"] = 0

    # Level n to level n+1 BER
    for level in range(num_levels - 1):
        upper_bound = target_ranges[level*2 + 1]
        level_data = groups[level]
        if len(level_data) > 0:
            ber_values[f"State{level} to State{level+1}"] = round(
                np.sum(level_data > upper_bound) / len(level_data) * 1e6
            )
        else:
            ber_values[f"State{level} to State{level+1}"] = 0

    return groups, groups_stats, real_selected_groups, ber_values
    
import itertools

def get_column_widths(table_data):
    """
    Calculate column widths based on the content length of each column, aiming to
    ensure all content, especially in the first row, fits well.
    Additionally, print the width of each cell and the maximum width for each column.
    """

    max_widths = []
    for column in zip(*table_data):
        cell_widths = [len(str(item)) for item in column]  # Calculate width for each cell in the column
        max_width = max(cell_widths)
        max_widths.append(max_width)
        #print(f"Cell widths in column: {cell_widths} -> Max width: {max_width}")
        
    #print("Max widths for all columns:", max_widths)
    
    def adjusted_length(item):
        # Count special characters and capital letters as 2
        return sum(2 if (not char.islower()) else 1 for char in str(item))
    
    max_widths = [max(adjusted_length(item) for item in column) for column in zip(*table_data)]
    #print(max_widths)
    
    # Normalize widths by the length of the longest cell to get relative sizes.
    max_total_width = sum(max_widths)
    column_widths = [width / max_total_width for width in max_widths]
    
    return column_widths

def plot_average_values_table(avg_values, table_names, selected_groups, base_figsize=(1, 2)):
    try:
        # Find the maximum value in each column and create a list of tuples
        table_data_list = []
        for i, (table_name, values) in enumerate(zip(table_names, avg_values)):
            max_value = max(values)  # Find maximum value in this column
            table_data_list.append((table_name, max_value, values))
        
        # Sort table data based on maximum values (descending order)
        table_data_list.sort(key=lambda x: x[1], reverse=True)
        
        # Unpack the sorted data
        table_names = [item[0] for item in table_data_list]
        avg_values = [item[2] for item in table_data_list]

        # Build table data with states as the first row and table names as the first column
        header = ["Table Name"] + [f"State {group}" for group in selected_groups]
        table_data = [header]  # Initialize table with header

        column_data = [[] for _ in range(len(selected_groups))]  # Store data per state for column stats

        for i, (table_name, values) in enumerate(zip(table_names, avg_values)):
            row = [f"{table_name}"]
            row_data = []

            for j, avg in enumerate(values):
                row.append(f"{avg:.2f}")
                row_data.append(avg)
                column_data[j].append(avg)

            table_data.append(row)

        # Calculate and append column averages and standard deviations
        footer_avg = ["Col Avg"]
        footer_std = ["Col Std Dev"]
        for col in column_data:
            avg = np.mean(col)
            std = np.std(col)
            footer_avg.append(f"{avg:.2f}")
            footer_std.append(f"{std:.2f}")

        table_data.append(footer_avg)
        table_data.append(footer_std)

        # Adjust figure size based on the number of rows and columns
        num_columns = len(table_data[0])
        num_rows = len(table_data)

        fig_width = max(num_columns * 1.5, 15)
        fig_height = max(num_rows * 0.5, 10)
        
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        ax.set_title('Averages')

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_std_values_table(std_values, table_names, selected_groups, base_figsize=(1, 2)):
    try:
        # Organize the data in the same way as in plot_average_values_table
        table_data_list = []
        for i, (table_name, values) in enumerate(zip(table_names, std_values)):
            max_value = max(values)  # Find maximum value in this column
            table_data_list.append((table_name, max_value, values))

        # Sort table data based on maximum values (descending order)
        table_data_list.sort(key=lambda x: x[1], reverse=True)

        # Unpack the sorted data
        table_names = [item[0] for item in table_data_list]
        std_values = [item[2] for item in table_data_list]

        # Build table data with states as the first row and table names as the first column
        header = ["Table Name"] + [f"State {group}" for group in selected_groups]
        table_data = [header]  # Initialize table with header

        column_data = [[] for _ in range(len(selected_groups))]  # Store data per state for column stats

        for i, (table_name, values) in enumerate(zip(table_names, std_values)):
            row = [f"{table_name}"]
            row_data = []

            for j, std in enumerate(values):
                row.append(f"{std:.2f}")
                row_data.append(std)
                column_data[j].append(std)

            table_data.append(row)

        # Calculate and append column averages and standard deviations
        footer_avg = ["Col Avg"]
        footer_std = ["Col Std Dev"]
        for col in column_data:
            avg = np.mean(col)
            std = np.std(col)
            footer_avg.append(f"{avg:.2f}")
            footer_std.append(f"{std:.2f}")

        table_data.append(footer_avg)
        table_data.append(footer_std)

        # Adjust figure size based on the number of rows and columns
        num_columns = len(table_data[0])
        num_rows = len(table_data)

        fig_width = max(num_columns * 1.5, 15)
        fig_height = max(num_rows * 0.5, 10)

        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        ax.set_title('Standard Deviations')

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_pass_range_ber_table(pass_range_ber_values, table_names, selected_groups, figsize=(15, 10)):
    try:
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Create table header
        header = ["State"] + [f"{table_name}" for table_name in table_names] + ["Row Avg", "Row Std Dev"]
        table_data = [header]

        column_data = [[] for _ in table_names]

        # Fill table data with pass_range_ber_values
        for i, group in enumerate(selected_groups):
            row = [f"State {group}"]
            row_data = []

            for j, table_pass_ber in enumerate(pass_range_ber_values):
                pass_ber = table_pass_ber[i]
                row += [f"{pass_ber:.0f}"]  # Append pass_range_ber value
                row_data.append(pass_ber)
                column_data[j].append(pass_ber)

            # Calculate and append row average and standard deviation
            row_avg = np.mean(row_data)
            row_std = np.std(row_data)
            row += [f"{row_avg:.0f}", f"{row_std:.0f}"]

            table_data.append(row)

        # Calculate and append column averages and standard deviations
        col_avgs = [np.mean(col) for col in column_data]
        col_stds = [np.std(col) for col in column_data]
        table_data.append(["Col Avg"] + [f"{avg:.0f}" for avg in col_avgs] + ["-", "-"])
        table_data.append(["Col Std Dev"] + [f"{std:.0f}" for std in col_stds] + ["-", "-"])

        column_widths = get_column_widths(table_data)

        # Create the table and set font size and scaling
        table = ax.table(cellText=table_data, loc='center', colWidths=column_widths, cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        ax.set_title('Pass Range BER Table')

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_colormap(data, title, figsize=(30, 15), g_range=None):
    # Create a new figure instance for this plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    try:
        if g_range is None:
            g_range = [0, np.max(data)]
        
        # Set minimum and maximum values for the color scale
        vmin, vmax = g_range
        cax = ax.imshow(data, cmap=plt.cm.viridis, origin="lower", vmin=vmin, vmax=vmax)
        fig.colorbar(cax)
        ax.set_title(title, fontsize=12)
        
        # Save the figure to a buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_colormap_magnified(data, title, figsize=(30, 15), g_range=None):
    # Create a new figure instance for this plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    try:
        if g_range is None:
            g_range = [0, np.max(data)]
        
        # Set minimum and maximum values for the color scale
        vmin, vmax = g_range
        cax = ax.imshow(data, cmap=plt.cm.viridis, origin="lower", vmin=vmin, vmax=vmax, aspect=0.1)
        fig.colorbar(cax)
        ax.set_title(title, fontsize=12)

        # Save the figure to a buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def get_full_table_data(table_name, database_name):
    connection = create_connection(database_name)
    query = f"SELECT * FROM `{table_name}`"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    close_connection()  # Make sure to pass the connection object to properly close it

    # Assuming the data is structured as a list of tuples, where each tuple represents a row in the table
    data_matrix = np.array(data)
    data_matrix = data_matrix.astype(float)
    data_matrix[data_matrix == 0] = 0.001   #0 is not working, why?
    
    # Get the size (shape) of the matrix
    data_matrix_size = data_matrix.shape

    return data_matrix, data_matrix_size
    
def plot_ber_tables(ber_results):
    # Extract unique table names and state transitions
    table_names = sorted(set(entry[0] for entry in ber_results))
    state_transitions = sorted(set(entry[1] for entry in ber_results))

    # Organize BER data
    ber_data = {}
    for entry in ber_results:
        table_name = entry[0]
        state_transition = entry[1]
        sigma = entry[2]
        ppm_ber = entry[4]
        uS_value = entry[5]
        hlyv_rounded = entry[6]

        if ppm_ber is not None:
            if table_name not in ber_data:
                ber_data[table_name] = {}
            ber_data[table_name][state_transition] = {
                'sigma': sigma,
                'ppm_ber': ppm_ber,
                'uS_value': uS_value,
                'hlyv_rounded': hlyv_rounded
            }

    # Calculate max BER per table
    max_ber_per_table = {
        table_name: max(transitions[t_data]['ppm_ber'] 
                       for t_data in transitions)
        for table_name, transitions in ber_data.items()
    }

    # Sort table names and create threshold-based lists
    sorted_table_names = sorted(max_ber_per_table, key=max_ber_per_table.get, reverse=False)
    sorted_tables_by_threshold = {
        threshold: [table for table in sorted_table_names 
                   if max_ber_per_table[table] <= threshold]
        for threshold in [100, 200, 500, 1000]
    }

    # Prepare headers and data structures
    headers = ["State/Transition"] + sorted_table_names + ["Row Avg"]
    data_structures = {
        'sigma': [headers],
        'ppm': [headers],
        'uS': [headers],
        'additional': [headers]
    }

    # Build data rows
    for state_transition in state_transitions:
        rows = {key: [state_transition] for key in data_structures}
        row_values = {key: [] for key in data_structures}

        for table_name in sorted_table_names:
            data = ber_data.get(table_name, {}).get(state_transition)
            if data:
                # Format and append values
                for key, (value, precision) in [
                    ('sigma', (data['sigma'], 4)),
                    ('ppm', (data['ppm_ber'], 0)),
                    ('uS', (data['uS_value'], 0)),
                    ('additional', (data['hlyv_rounded'], 4))
                ]:
                    formatted_value = f"{value:.{precision}f}" if value is not None else "N/A"
                    rows[key].append(formatted_value)
                    if value is not None:
                        row_values[key].append(value)
            else:
                for key in data_structures:
                    rows[key].append("N/A")

        # Add row averages
        for key in data_structures:
            values = row_values[key]
            avg = f"{np.mean(values):.{4 if key in ['sigma', 'additional'] else 0}f}" if values else "N/A"
            rows[key].append(avg)
            data_structures[key].append(rows[key])

    # Transpose data and generate plots
    images = {}
    titles = {
        'sigma': "Sigma Values at Intersection",
        'ppm': "BER PPM",
        'uS': "BER at Windows = 2",
        'additional': "Y Values at Windows = 2"
    }

    for key, title in titles.items():
        transposed_data = [list(x) for x in zip(*data_structures[key])]
        images[key] = plot_table(transposed_data, title, transpose=False)

    return (images['sigma'],
            images['ppm'],
            images['uS'],
            images['additional'],
            sorted_table_names,
            sorted_tables_by_threshold.get(100, []),
            sorted_tables_by_threshold.get(200, []),
            sorted_tables_by_threshold.get(500, []),
            sorted_tables_by_threshold.get(1000, []))

def plot_table(data, title, transpose=False):
    """
    Plots a table from data and returns it as a base64-encoded image.
    Updated for thread safety and proper resource management.
    """
    if transpose:
        # Transpose the data
        data = [list(x) for x in zip(*data)]

    num_columns = len(data[0])
    num_rows = len(data)

    # Dynamically adjust the figure size to match other tables
    fig_width = max(num_columns * 1.5, 15)  # Changed from 12 to 15 to match other tables
    fig_height = max(num_rows * 0.5, 10)    # Changed from 6 to 10 to match other tables
    
    # Create a new figure instance for this plot
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    
    try:
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=data, loc='center', cellLoc='center')

        # Set font size and scaling to match other tables
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)  # Changed from 1.5 to match other tables

        ax.set_title(title, fontsize=12)

        # Save the figure to a buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def get_colors(num_colors):
    """Generate a colormap and return the colors for the specified number of items."""
    cmap = plt.get_cmap('viridis', num_colors)
    norm = mcolors.Normalize(vmin=0, vmax=num_colors - 1)
    return [cmap(norm(i)) for i in range(num_colors)]

def plot_miao(combined_data, base_figsize=(1, 2)):
    # Determine dimensions
    table_names = list(combined_data.keys())
    ber_levels = list(next(iter(combined_data.values())).keys())
    num_columns = len(table_names) + 1
    num_rows = len(ber_levels) + 1

    # Dynamically set the figure size
    fig_width = max(num_columns * 1.5, 15)
    fig_height = max(num_rows * 0.5, 10)
    
    # Create a new figure instance for this plot
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    
    try:
        ax.axis('off')

        # Create table header
        header = ["State/Transition"] + table_names
        table_data = [header]

        # Prepare data rows
        for level in ber_levels:
            row = [level]
            for table_name in table_names:
                row.append(f"{combined_data[table_name][level]:,.0f}")
            table_data.append(row)

        # Create the table and set properties
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        ax.set_title('Pass Range BER', fontsize=12)

        # Save to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_individual_points_map(correlation_analysis, table_dimensions):
    """Plot individual outlier points on a 2D map."""
    try:
        # Debug prints
        print("Received correlation_analysis:", correlation_analysis)
        print("Type of correlation_analysis:", type(correlation_analysis))
        print("Table dimensions:", table_dimensions)

        # Handle None case
        if correlation_analysis is None:
            print("Correlation analysis is None")
            return None

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        # Keep track of annotated coordinates
        annotated_coords = set()
        
        # Handle both dictionary and list formats
        if isinstance(correlation_analysis, dict):
            exact_matches = correlation_analysis.get('exact_matches', [])
            region_clusters = correlation_analysis.get('region_clusters', [])
            print(f"Found {len(exact_matches)} exact matches and {len(region_clusters)} region clusters")
            
            # Plot exact matches
            for match in exact_matches:
                if not isinstance(match, dict) or 'coordinate' not in match or len(match['coordinate']) != 2:
                    print(f"Skipping invalid match: {match}")
                    continue
                    
                row, col = match['coordinate']
                coord_tuple = (row, col)
                
                ax.scatter(col, row, color='red', s=50, alpha=0.6)
                if coord_tuple not in annotated_coords:
                    coord_str = f"({row}, {col})"
                    ax.annotate(coord_str, (col, row), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8)
                    annotated_coords.add(coord_tuple)

            # Plot region clusters with different colors
            colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(region_clusters))))
            for cluster, color in zip(region_clusters, colors):
                if not isinstance(cluster, dict) or 'outliers' not in cluster:
                    print(f"Skipping invalid cluster: {cluster}")
                    continue
                    
                outliers = cluster['outliers']
                for outlier in outliers:
                    if not isinstance(outlier, dict) or 'coordinates' not in outlier or len(outlier['coordinates']) != 2:
                        print(f"Skipping invalid outlier: {outlier}")
                        continue
                        
                    row, col = outlier['coordinates']
                    coord_tuple = (row, col)
                    
                    ax.scatter(col, row, color=color, s=50, alpha=0.6)
                    if coord_tuple not in annotated_coords:
                        coord_str = f"({row}, {col})"
                        ax.annotate(coord_str, (col, row), 
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=8)
                        annotated_coords.add(coord_tuple)
                               
        elif isinstance(correlation_analysis, list):
            # Handle direct list of outliers
            print(f"Found {len(correlation_analysis)} direct outliers")
            for outlier in correlation_analysis:
                if not isinstance(outlier, dict) or 'coordinates' not in outlier or len(outlier['coordinates']) != 2:
                    print(f"Skipping invalid outlier: {outlier}")
                    continue
                    
                row, col = outlier['coordinates']
                coord_tuple = (row, col)
                
                ax.scatter(col, row, color='red', s=50, alpha=0.6)
                if coord_tuple not in annotated_coords:
                    coord_str = f"({row}, {col})"
                    ax.annotate(coord_str, (col, row), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8)
                    annotated_coords.add(coord_tuple)
        else:
            print(f"Unsupported correlation_analysis type: {type(correlation_analysis)}")
            return None

        # Set axis limits and labels
        ax.set_xlim(-1, table_dimensions[1])
        ax.set_ylim(-1, table_dimensions[0])
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('Outlier Points Map (Values < 50)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Convert plot to base64 string
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        print("Successfully generated plot")
        print(f"Total unique coordinates annotated: {len(annotated_coords)}")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error in plot_individual_points_map: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if 'buf' in locals():
            buf.close()