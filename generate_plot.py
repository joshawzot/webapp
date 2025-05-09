from tools_for_plots import *
import io
import base64
import pandas as pd
import re
from conductance_calculator import convert_table_to_conductance, convert_table_to_linear

def get_group_data_1124(table_name, selected_groups, database_name, pattern_file_array):
    connection = create_connection(database_name)
    query = f"SELECT * FROM {table_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
        
    # Convert fetched data to a NumPy array for easier manipulation
    data_np = np.array(data, dtype=float)  # Ensure data is converted to float
    
    # Replace zeros with a small value to avoid issues - use np.where for safer comparison
    data_np = np.where(data_np == 0, 0.001, data_np)

    # if np.mean(data_np) < 1:
    #     data_np = data_np * 1e6

    groups = []
    groups_stats = []  # List to store statistics for each group
    group_idx_to_position = {}

    # Ensure that pattern_file_array has the same shape as data_np
    if pattern_file_array.shape != data_np.shape:
        raise ValueError("pattern_file_array must have the same shape as the data array.")

    unique_groups = np.unique(pattern_file_array)
    group_indices = unique_groups.tolist()
    print("group_indices:", group_indices)  # e.g., [0, 1, 2, 3]

    for group_idx in group_indices:
        if group_idx in selected_groups:
            # Get the mask where pattern_file_array equals group_idx
            group_mask = np.equal(pattern_file_array, group_idx)  # Use np.equal instead of == for better array handling
            group_data = data_np[group_mask]
            
            # Filter out negative values - use np.greater_equal for safer comparison
            positive_group_data = group_data[np.greater_equal(group_data, 0)]

            group_idx_to_position[group_idx] = len(groups)
            groups.append(positive_group_data)

            # Calculate statistics for the positive values
            if len(positive_group_data) > 0:
                average = round(np.mean(positive_group_data), 2)
                std_dev = round(np.std(positive_group_data), 2)
                outlier_condition = np.abs(positive_group_data - average) > 2.698 * std_dev
                outlier_percentage = round(np.sum(outlier_condition) / len(positive_group_data) * 100, 2)
                groups_stats.append((table_name, group_idx, average, std_dev, outlier_percentage))
            else:
                print(f"State {group_idx} has no positive values for analysis.")

    # Sort groups_stats by average value in ascending order
    groups_stats.sort(key=lambda x: x[2])

    # Rearrange groups according to the sorted order
    sorted_groups = []
    for stats in groups_stats:
        table_name, group_idx, average, std_dev, outlier_percentage = stats
        position = group_idx_to_position[group_idx]
        sorted_groups.append(groups[position])

    groups = sorted_groups

    # Extract sorted group indices from groups_stats (keep original indices)
    group_indices = [i[1] for i in groups_stats]  # Now correctly holds original group indices
    print("group_indices____", group_indices)

    close_connection()

    return groups, groups_stats, group_indices

def get_group_data_1124_2(target_ranges, table_name, selected_groups, database_name, pattern_file_array):
    connection = create_connection(database_name)
    query = f"SELECT * FROM {table_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()

    data_np = np.array(data, dtype=float)  # Ensure data is converted to float
    
    # Replace zeros with a small value to avoid issues - use np.where for safer comparison
    data_np = np.where(data_np == 0, 0.001, data_np)  

    # if np.mean(data_np) < 1:
    #     data_np = data_np * 1e6

    # Ensure that pattern_file_array has the same shape as data_np
    if pattern_file_array.shape != data_np.shape:
        raise ValueError("pattern_file_array must have the same shape as the data array.")

    groups = []
    groups_stats = []  # List to store statistics for each group
    real_selected_groups = []
    group_idx_to_position = {}
    count = 0

    unique_groups = np.unique(pattern_file_array)
    group_indices = unique_groups.tolist()

    for group_idx in group_indices:
        if group_idx in selected_groups:
            real_selected_groups.append(group_idx)

            try:
                # Create a mask where pattern_file_array equals group_idx
                group_mask = np.equal(pattern_file_array, group_idx)  # Use np.equal instead of == for better comparison
                group = data_np[group_mask]
                flattened_group = group.flatten()

                # Filter out negative values - use np.greater_equal for safer comparison
                positive_flattened_group = flattened_group[np.greater_equal(flattened_group, 0)]
                groups.append(positive_flattened_group)

                # Calculate statistics for the positive values
                if len(group) > 0:
                    average = round(np.mean(group), 2)
                    std_dev = round(np.std(group), 2)

                    # Get the target range for this group
                    lower_bound, upper_bound = target_ranges[count * 2], target_ranges[count * 2 + 1]

                    # Calculate the BER (ppm value of data outside the target range)
                    # Use np.logical_or with np.less/np.greater for safer comparison
                    out_of_range_condition = np.logical_or(
                        np.less(group, lower_bound),
                        np.greater(group, upper_bound)
                    )
                    out_of_range_data = group[out_of_range_condition]
                    ber_value = round(len(out_of_range_data) / len(group) * 1e6)  # Calculate ppm

                    # Store statistics including BER value and target ranges
                    # Use np.greater instead of > for safer comparison
                    outlier_condition = np.greater(np.abs(group - average), 2.698 * std_dev)
                    outlier_percentage = round(
                        np.sum(outlier_condition) / len(group) * 100, 0
                    )

                    groups_stats.append((
                        table_name, group_idx, average, std_dev, outlier_percentage, ber_value, lower_bound, upper_bound
                    ))
                    group_idx_to_position[group_idx] = len(groups) - 1  # Map group_idx to index in groups list
                    count += 1
                else:
                    print(f"State {group_idx} has no positive values for analysis.")
            except IndexError as e:
                print(f"Error accessing data slice: {e}")

    close_connection()

    # Sort groups_stats by average value in ascending order
    groups_stats.sort(key=lambda x: x[2])  # Sort by average value

    # Reassign group_idx to reflect the sorted order and rearrange groups accordingly
    sorted_groups = []
    new_groups_stats = []
    for idx, stats in enumerate(groups_stats):
        table_name, original_group_idx, average, std_dev, outlier_percentage, ber_value, lower_bound, upper_bound = stats
        new_group_idx = idx  # Assign new group index based on sorted order
        new_groups_stats.append((
            table_name, new_group_idx, average, std_dev, outlier_percentage, ber_value, lower_bound, upper_bound
        ))
        # Get the group data corresponding to original_group_idx
        position = group_idx_to_position[original_group_idx]
        sorted_groups.append(groups[position])

    groups_stats = new_groups_stats
    groups = sorted_groups

    # Update real_selected_groups to match the new group indices
    real_selected_groups = [stats[1] for stats in groups_stats]  # This will be [0, 1, 2, ...]

    # Reconstruct target_ranges from the sorted groups_stats
    target_ranges = []
    for stats in groups_stats:
        lower_bound, upper_bound = stats[6], stats[7]
        target_ranges.extend([lower_bound, upper_bound])

    # Calculate BER values for different levels and transitions
    num_levels = len(groups)
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
            ber_values[f"State{level} to State{level-1}"] = 0

    # Level n to level n+1 BER
    for level in range(num_levels - 1):
        upper_bound = target_ranges[level * 2 + 1]
        level_data = groups[level]
        if len(level_data) > 0:
            ber_values[f"State{level} to State{level+1}"] = round(
                np.sum(level_data > upper_bound) / len(level_data) * 1e6
            )
        else:
            ber_values[f"State{level} to State{level+1}"] = 0

    return groups, groups_stats, real_selected_groups, ber_values

def reorder_tables_fuxi(table_names):
    # Function to extract the numeric parts and consider the rest as non-numeric
    def extract_parts(s):
        parts = re.split(r'(\d+)', s)
        non_numeric_parts = ''.join(part for i, part in enumerate(parts) if i % 2 == 0)
        numeric_parts = tuple(int(part) for i, part in enumerate(parts) if i % 2 != 0)
        return (non_numeric_parts, numeric_parts)

    # Sort the list using the non-numeric part and numeric parts as a tuple
    return sorted(table_names, key=extract_parts)
    
def extract_number_from_table_name(table_name):
    """
    Extract the number after 'io' and before the next '_' in the table_name.
    """
    match = re.search(r'io(\d+)_', table_name)
    if match:
        return match.group(1)
    else:
        return table_name

def analyze_coordinate_correlations(outlier_coordinates):
    """Analyze correlations between outlier coordinates across different tables."""
    try:
        if not outlier_coordinates or not isinstance(outlier_coordinates, list):
            print("No outlier coordinates to analyze or invalid input type")
            return {
                'exact_matches': [],
                'region_clusters': [],
                'summary': {
                    'total_outliers': 0,
                    'unique_coordinates': 0,
                    'coordinates_in_multiple_tables': 0,
                    'number_of_clusters': 0
                }
            }

        # Group outliers by coordinates
        coord_map = {}
        for outlier in outlier_coordinates:
            try:
                # Ensure outlier is a dictionary with required keys
                if not isinstance(outlier, dict) or 'coordinates' not in outlier:
                    print(f"Invalid outlier format: {outlier}")
                    continue
                
                # Get coordinates and ensure they are a list/tuple of 2 integers
                coords = outlier.get('coordinates', [])
                if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                    print(f"Invalid coordinates format: {coords}")
                    continue
                
                # Convert coordinates to tuple for dictionary key
                try:
                    coord = (int(coords[0]), int(coords[1]))
                except (TypeError, ValueError) as e:
                    print(f"Error converting coordinates to integers: {e}")
                    continue
                
                # Store in coord_map
                if coord not in coord_map:
                    coord_map[coord] = []
                coord_map[coord].append(outlier)
            except Exception as e:
                print(f"Error processing outlier: {e}")
                continue
        
        # Initialize results structure
        correlation_results = {
            'exact_matches': [],
            'region_clusters': [],
            'summary': {}
        }
        
        # Find exact matches
        for coord, outliers in coord_map.items():
            if len(outliers) > 1:
                try:
                    match_entry = {
                        'coordinate': list(coord),  # Convert tuple to list
                        'tables': [],
                        'values': [],
                    }
                    
                    for o in outliers:
                        try:
                            match_entry['tables'].append(str(o.get('table', '')))
                            match_entry['values'].append(float(o.get('value', 0.0)))
                        except (ValueError, TypeError) as e:
                            print(f"Error processing outlier values: {e}")
                            continue
                    
                    if match_entry['tables']:
                        correlation_results['exact_matches'].append(match_entry)
                except Exception as e:
                    print(f"Error creating match entry: {e}")
                    continue
        
        # Find nearby coordinates (within 5 units)
        def distance(coord1, coord2):
            try:
                return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
            except (TypeError, IndexError):
                return float('inf')
        
        # Group coordinates into clusters
        coords = list(coord_map.keys())
        clusters = []
        used_coords = set()
        
        for i, coord1 in enumerate(coords):
            if coord1 in used_coords:
                continue
            
            cluster = {coord1}
            used_coords.add(coord1)
            
            # Find all coordinates within 5 units of this coordinate
            for coord2 in coords[i+1:]:
                if coord2 not in used_coords and distance(coord1, coord2) <= 5:
                    cluster.add(coord2)
                    used_coords.add(coord2)
            
            if len(cluster) > 1:
                try:
                    cluster_outliers = []
                    for coord in cluster:
                        for o in coord_map[coord]:
                            try:
                                cluster_outliers.append({
                                    'table': str(o.get('table', '')),
                                    'coordinates': list(coord),  # Convert tuple to list
                                    'value': float(o.get('value', 0.0))
                                })
                            except (ValueError, TypeError, KeyError) as e:
                                print(f"Error processing cluster outlier: {e}")
                                continue
                    
                    if cluster_outliers:
                        correlation_results['region_clusters'].append({
                            'coordinates': [list(c) for c in cluster],  # Convert tuples to lists
                            'outliers': cluster_outliers
                        })
                except Exception as e:
                    print(f"Error creating cluster: {e}")
                    continue
        
        # Generate summary
        correlation_results['summary'] = {
            'total_outliers': len(outlier_coordinates),
            'unique_coordinates': len(coord_map),
            'coordinates_in_multiple_tables': len(correlation_results['exact_matches']),
            'number_of_clusters': len(correlation_results['region_clusters'])
        }
        
        return correlation_results
    
    except Exception as e:
        print(f"Error in analyze_coordinate_correlations: {e}")
        return {
            'exact_matches': [],
            'region_clusters': [],
            'summary': {
                'total_outliers': len(outlier_coordinates) if isinstance(outlier_coordinates, list) else 0,
                'unique_coordinates': 0,
                'coordinates_in_multiple_tables': 0,
                'number_of_clusters': 0
            }
        }

def calculate_sigma_distances(data, target_values, table_names):
    print("Entering calculate_sigma_distances")
    print("data length:", len(data))
    print("target_values:", target_values)
    print("table_names:", table_names)
    
    sigma_distances = {}
    sigma_points = [-4, -3, -2, -1, 0, 1, 2, 3, 4]  # Sigma points to analyze
    
    for table_idx, (table_name, table_data) in enumerate(zip(table_names, data)):
        print(f"Processing table {table_name}")
        sigma_distances[table_name] = []
        
        for state_idx, state_data in enumerate(table_data):
            print(f"Processing state {state_idx}")
            if state_idx < len(target_values):  # Only process if we have a target value
                target = target_values[state_idx]
                mean = np.mean(state_data)
                std = np.std(state_data)
                
                print(f"State {state_idx} stats:")
                print(f"  Target: {target}")
                print(f"  Mean: {mean}")
                print(f"  Std: {std}")
                
                # Calculate distances at each sigma point
                distances = []
                for sigma in sigma_points:
                    point = mean + (sigma * std)
                    distance = point - target
                    distances.append(distance)
                
                sigma_distances[table_name].append(distances)
                print(f"  Distances calculated: {distances}")
    
    print("Final sigma_distances:", sigma_distances)
    return sigma_distances

def generate_plot(table_names, database_name, form_data):
    print("form_data:", form_data)
    color_map_flag = form_data['color_map_flag']  # This is now a boolean
    outlier_analysis_flag = form_data.get('outlier_analysis_flag', False)  # Default to False if not provided
    target_values = form_data.get('target_values', [])  # Get target values from form_data
    custom_division = form_data.get('custom_division', False)  # Get custom_division flag, default to False
    
    # Get the user-defined target_x_diff value with a default of 2
    target_x_diff = float(form_data.get('target_x_diff', 2))  # Get target_x_diff from form_data
    print("target_x_diff:", target_x_diff)
    #target_x_diff = 0
    # Check if we should use conductance values or linear conversion
    using_conductance = form_data.get('using_conductance', False)
    using_linear_conversion = form_data.get('using_linear_conversion', False)
    conductance_params = form_data.get('conductance_params', {})
    
    # Initialize sigma_distances and num_states at the start
    sigma_distances = {}
    num_states = 0
    
    print("color_map_flag:", color_map_flag)
    print("outlier_analysis_flag:", outlier_analysis_flag)
    print("target_values:", target_values)  # Print target values for debugging
    print("custom_division:", custom_division)  # Print custom_division for debugging
    print("using_conductance:", using_conductance)  # Print using_conductance for debugging
    print("using_linear_conversion:", using_linear_conversion)  # Print using_linear_conversion for debugging

    print("table_names:", table_names)
    table_names = reorder_tables_fuxi(table_names)
    print("reordered_table_names:", table_names)

    selected_groups = form_data.get('selected_groups', "")
    print("selected_groups:", selected_groups)
    
    # Initialize variables for outlier analysis
    outlier_coordinates = []
    correlation_analysis = None
    cluster_map = None
    
    if form_data['state_pattern_type'] == 'predefined':
        # Define the path to your state pattern files directory
        state_pattern = form_data.get('state_pattern')
        print("state_pattern:", state_pattern)
        # Define a dictionary to map state patterns to their file paths
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

        # Fetch the file path based on the state pattern using a dictionary lookup
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
    elif form_data['state_pattern_type'] == '1D':
        state_pattern = None
        number_of_states = form_data.get('number_of_states', "")
        print("number_of_states:", number_of_states)

    # Retrieve target_ranges correctly
    pass_range = form_data.get('pass_range_predefined') or form_data.get('pass_range_1D')
    if pass_range == "custom":
        print("A")
        target_ranges = form_data.get('custom_pass_range_predefined', "") or form_data.get('custom_pass_range_1D', "")
    else:
        print("B")
        target_ranges = pass_range

    print("target_ranges:", target_ranges)
    target_ranges = [float(x) for x in target_ranges.split(',') if x.replace('.', '', 1).isdigit()]
    print("target_ranges:", target_ranges)

    # Check if target_ranges has values
    if target_ranges:
        print("Target ranges have values:", target_ranges)
        target_range_flag = 1
    else:
        print("Target ranges are empty or not provided.")
        target_range_flag = 0

    # Initialize an empty list to hold the encoded plots
    encoded_plots = []
    group_data = []
    colors = get_colors(len(table_names))
    avg_values = []
    std_values = []
    miao_ber = []
    sub_array_size = []

    # Compute the global min and max values among all data matrices
    data_matrices = []
    for table_name in table_names:
        data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
        
        # Apply conductance conversion if enabled
        if using_conductance and conductance_params:
            print(f"Converting table {table_name} to conductance values")
            data_matrix = convert_table_to_conductance(data_matrix, conductance_params)
        # Apply linear conversion if enabled
        elif using_linear_conversion:
            print(f"Converting table {table_name} using linear conversion (0-63 → 60-170)")
            data_matrix = convert_table_to_linear(data_matrix)
            
        data_matrices.append((table_name, data_matrix))
    
    # Ensure all data matrices are converted to float
    data_matrices = [(label, data_matrix.astype(float)) for label, data_matrix in data_matrices]

    print("min")
    global_min = min(np.min(data_matrix.astype(float)) for _, data_matrix in data_matrices)
    global_max = max(np.max(data_matrix.astype(float)) for _, data_matrix in data_matrices)
    g_range = (global_min, global_max)
    print("min")

    for i, table_name in enumerate(table_names):
        # Use the already processed data matrix
        data_matrix = data_matrices[i][1]
        
        if target_range_flag == 0:
            if form_data['state_pattern_type'] == '1D':
                # Modify to use the data matrix directly
                groups, stats, selected_groups = get_group_data_new_from_matrix(
                    data_matrix, selected_groups, number_of_states, custom_division)
            elif form_data['state_pattern_type'] == 'predefined':
                # Modify to use the data matrix directly
                groups, stats, selected_groups = get_group_data_from_matrix(
                    data_matrix, selected_groups, pattern_file_array)
        elif target_range_flag == 1:
            if form_data['state_pattern_type'] == '1D':
                # Modify to use the data matrix directly
                groups, stats, selected_groups, table_miao_ber = get_group_data_latest_from_matrix(
                    target_ranges, data_matrix, selected_groups, number_of_states, custom_division)
            elif form_data['state_pattern_type'] == 'predefined':
                # Modify to use the data matrix directly
                groups, stats, selected_groups, table_miao_ber = get_group_data_1124_2_from_matrix(
                    target_ranges, data_matrix, selected_groups, pattern_file_array)
            miao_ber.append(table_miao_ber)

        # Extract average and standard deviation values for each selected group
        table_avg_values = [stat[2] for stat in stats]  # Index 2 is average
        table_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation

        group_data.append(groups)
        avg_values.append(table_avg_values)
        std_values.append(table_std_values)

        # Before calculating sigma distances, add debug prints
        print("About to check target_values condition")
        print("target_values is:", target_values)
        print("Is target_values truthy?", bool(target_values))
        print("group_data structure:", [len(group) for group in group_data])
        
        if target_values:
            print("Inside target_values condition")
            print("group_data length:", len(group_data))
            sigma_distances = calculate_sigma_distances(group_data, target_values, table_names)
            num_states = len(target_values)
            print("Calculated sigma distances:", sigma_distances)
        else:
            print("target_values condition was False")

    print("equal")
    def combine_data(table_names, miao_ber):
        # Check that the lengths of both lists match
        if len(table_names) != len(miao_ber):
            raise ValueError("Length of table_names and miao_ber must be the same")

        # Combine table names with BER data
        combined = {}
        for table_name, ber_data in zip(table_names, miao_ber):
            combined[table_name] = ber_data

        return combined

    if target_range_flag == 1:
        miao_ber = combine_data(table_names, miao_ber)
        print(miao_ber)

    # Plot the color maps using the shared color scale if color_map_flag is True
    if color_map_flag:
        for table_name, data_matrix in data_matrices:
            if state_pattern in ("1296x64_rowbar_4states", "1296x64_Adrien_random_4states", "1296x64_1state"):
                encoded_plots.append(plot_colormap_magnified(
                    data_matrix, title=f"Colormap for {table_name}", g_range=g_range))
            else:
                encoded_plots.append(plot_colormap(
                    data_matrix, title=f"Colormap for {table_name}", g_range=g_range))

    # Generate plots for individual tables using original 'table_names'
    encoded_plots.append(plot_boxplot(group_data, table_names))
    #encoded_plots.append(plot_histogram(group_data, table_names, colors))

    encoded_plots.append(plot_average_values_table(avg_values, table_names, selected_groups))
    encoded_plots.append(plot_std_values_table(std_values, table_names, selected_groups))

    plot_data_sigma, plot_data_cdf, plot_data_interpo, ber_results, sigma_intersections = plot_transformed_cdf_2(group_data, table_names, selected_groups, colors, target_x_diff)
    encoded_plots.append(plot_data_sigma)
    encoded_plots.append(plot_data_cdf)
    encoded_plots.append(plot_data_interpo)

    # Create a table for sigma intersections
    sigma_points = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    sigma_table = {}
    for table_name in table_names:
        sigma_table[table_name] = sigma_intersections[table_name]

    if target_range_flag == 1:
        print("miao_ber:", miao_ber)
        encoded_plots.append(plot_miao(miao_ber))
    
    # Only perform outlier analysis if the flag is enabled and state_pattern_type is predefined
    if outlier_analysis_flag and form_data['state_pattern_type'] == 'predefined':
        try:
            # Process each table's data for outliers
            for table_idx, table_name in enumerate(table_names):
                try:
                    # Get the data matrix for this table
                    data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
                    if not isinstance(data_matrix_size, tuple) or len(data_matrix_size) != 2:
                        print(f"Warning: Invalid data_matrix_size for table {table_name}")
                        continue
                        
                    rows, cols = data_matrix_size
                    print(f"Table {table_name} dimensions: {rows}x{cols}")
                    
                    # Get the last group's data
                    if len(group_data) > 0 and len(group_data[table_idx]) > 0:
                        last_group_idx = len(selected_groups) - 1 if selected_groups else 0
                        if last_group_idx >= len(group_data[table_idx]):
                            print(f"Warning: last_group_idx {last_group_idx} exceeds group_data length")
                            continue
                            
                        last_group = np.array(group_data[table_idx][last_group_idx], dtype=float)
                        last_group = np.ravel(last_group)
                        
                        if len(last_group) == 0:
                            print(f"Warning: Empty last_group for table {table_name}")
                            continue
                            
                        # Find outliers (values < 50)
                        outlier_mask = (last_group < 50)  # Use direct value threshold
                        outlier_indices = np.where(outlier_mask)[0]
                        outlier_values = last_group[outlier_mask]
                        
                        print(f"Found {len(outlier_indices)} outliers (value < 50) in table {table_name}")
                        
                        # Process each outlier
                        for i, (idx, value) in enumerate(zip(outlier_indices, outlier_values)):
                            try:
                                # Convert linear index to 2D coordinates
                                # Invert the row calculation to ensure last level appears at higher row indices
                                row = (rows - 1) - (int(idx) // int(cols))  # Invert row calculation
                                col = int(idx) % int(cols)
                                
                                # Validate coordinates
                                if not (0 <= row < rows and 0 <= col < cols):
                                    print(f"Warning: Invalid coordinates ({row}, {col}) for dimensions {rows}x{cols}")
                                    continue
                                
                                # Create outlier entry with explicit type conversion
                                outlier_entry = {
                                    'table': str(table_name),
                                    'coordinates': [int(row), int(col)],
                                    'value': float(value)
                                }
                                outlier_coordinates.append(outlier_entry)
                            except Exception as e:
                                print(f"Error processing outlier at index {idx}: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error processing table {table_name}: {str(e)}")
                    continue

            # Sort outliers by value if we have any outliers
            if outlier_coordinates:
                outlier_coordinates.sort(key=lambda x: float(x.get('value', 0)), reverse=False)  # Sort by value, lowest first

            # Analyze correlations between outlier coordinates
            try:
                correlation_analysis = analyze_coordinate_correlations(outlier_coordinates)
                print("correlation_analysis:", correlation_analysis)
                # Generate cluster map if we have correlation analysis
                if correlation_analysis:
                    # Get the dimensions from the first table's data matrix
                    first_table_name = table_names[0]
                    data_matrix, data_matrix_size = get_full_table_data(first_table_name, database_name)
                    rows, cols = data_matrix_size
                    cluster_map = plot_individual_points_map(correlation_analysis, table_dimensions=(rows, cols))
                    print("cluster_map generated:", cluster_map is not None)
            except Exception as e:
                print(f"Error in correlation analysis: {str(e)}")
                correlation_analysis = {
                    'exact_matches': [],
                    'region_clusters': [],
                    'summary': {
                        'total_outliers': len(outlier_coordinates),
                        'unique_coordinates': 0,
                        'coordinates_in_multiple_tables': 0,
                        'number_of_clusters': 0
                    }
                }

        except Exception as e:
            print(f"Error in outlier analysis: {str(e)}")
            outlier_coordinates = []
            correlation_analysis = None

    if len(selected_groups) != 1:
        # Generate plots for BER results and get sorted table names
        (sigma_image,
         ppm_image,
         uS_image,
         additional_image,
         sorted_table_names,
         sorted_table_names_100ppm,
         sorted_table_names_200ppm,
         sorted_table_names_500ppm,
         sorted_table_names_1000ppm) = plot_ber_tables(ber_results, target_x_diff)

        # Since we now have a combined image, append it to the plots
        encoded_plots.append(ppm_image)

        # Now, process 'sorted_table_names' to get the extracted numbers
        def process_sorted_table_names(sorted_table_names_list):
            if sorted_table_names_list:
                return [extract_number_from_table_name(name) for name in sorted_table_names_list]
            else:
                return []

        sorted_table_names = process_sorted_table_names(sorted_table_names)
        sorted_table_names_100ppm = process_sorted_table_names(sorted_table_names_100ppm)
        sorted_table_names_200ppm = process_sorted_table_names(sorted_table_names_200ppm)
        sorted_table_names_500ppm = process_sorted_table_names(sorted_table_names_500ppm)
        sorted_table_names_1000ppm = process_sorted_table_names(sorted_table_names_1000ppm)

        # Add the following code to create best_32 and best_32_with_io
        if sorted_table_names:
            best_32 = sorted_table_names[:32]
            best_32_with_io = ['io' + str(n) for n in best_32]
        else:
            best_32 = []
            best_32_with_io = []

        # Return the plots and sorted table names
        return (encoded_plots,
                sorted_table_names,
                sorted_table_names_100ppm,
                sorted_table_names_200ppm,
                sorted_table_names_500ppm,
                sorted_table_names_1000ppm,
                best_32,
                best_32_with_io,
                outlier_coordinates if outlier_analysis_flag else [],  # Only return outlier coordinates if flag is True
                correlation_analysis if outlier_analysis_flag else None,  # Only return correlation analysis if flag is True
                cluster_map if outlier_analysis_flag else None,
                sigma_distances,
                num_states,
                table_names,
                sigma_table,  # Add sigma intersections table
                sigma_points)  # Add sigma points
    else:
        sorted_table_names = None  # Handle the case where there is only one selected group

    # Now, process 'sorted_table_names' to get the extracted numbers
    if sorted_table_names:
        sorted_table_names = [extract_number_from_table_name(name) for name in sorted_table_names]
    else:
        sorted_table_names = []

    # Add the following code to create best_32 and best_32_with_io
    if sorted_table_names:
        best_32 = sorted_table_names[:32]
        best_32_with_io = ['io' + str(n) for n in best_32]
    else:
        best_32 = []
        best_32_with_io = []
    
    return (encoded_plots,
            sorted_table_names,
            None,
            None,
            None,
            None,
            best_32,
            best_32_with_io,
            outlier_coordinates if outlier_analysis_flag else [],
            correlation_analysis if outlier_analysis_flag else None,
            cluster_map if outlier_analysis_flag else None,
            sigma_distances,
            num_states,
            table_names,
            sigma_table,  # Add sigma intersections table
            sigma_points)  # Add sigma points

# Add helper functions to work with matrices directly instead of fetching from database

def get_group_data_from_matrix(data_matrix, selected_groups, pattern_file_array):
    """Modified version of get_group_data_1124 that works with a data matrix directly."""
    # Replace zeros with a small value to avoid issues - use np.where for safer comparison
    data_np = np.where(data_matrix == 0, 0.001, data_matrix)

    groups = []
    groups_stats = []  # List to store statistics for each group
    group_idx_to_position = {}

    # Ensure that pattern_file_array has the same shape as data_np
    if pattern_file_array.shape != data_np.shape:
        raise ValueError("pattern_file_array must have the same shape as the data array.")

    unique_groups = np.unique(pattern_file_array)
    group_indices = unique_groups.tolist()
    print("group_indices:", group_indices)

    # Parse the selected_groups
    if selected_groups:
        try:
            if isinstance(selected_groups, str):
                selected_groups = [int(g) for g in selected_groups.split(',')]
            elif isinstance(selected_groups, list):
                selected_groups = [int(g) for g in selected_groups]
        except ValueError:
            selected_groups = group_indices
    else:
        selected_groups = group_indices

    # Filter out selected_groups that don't exist in the dataset
    selected_groups = [g for g in selected_groups if g in group_indices]

    for group_idx in selected_groups:
        # Find positions where pattern_file_array equals the current group index
        positions = np.where(pattern_file_array == group_idx)
        values = [data_np[pos] for pos in zip(positions[0], positions[1])]
        
        # Store the group data for later use
        groups.append(values)
        
        # Compute statistics for the group
        if values:
            min_val = np.min(values)
            max_val = np.max(values)
            avg_val = np.mean(values)
            std_val = np.std(values)
            groups_stats.append((min_val, max_val, avg_val, std_val))
        else:
            # Default values if the group has no data points
            groups_stats.append((0, 0, 0, 0))
            
        # Map group index to its position in the selected_groups list
        group_idx_to_position[group_idx] = len(groups) - 1

    return groups, groups_stats, selected_groups

def get_group_data_new_from_matrix(data_matrix, selected_groups, number_of_states, custom_division=False):
    """Modified version of get_group_data_new that works with a data matrix directly."""
    # Debug prints to diagnose custom_division usage
    print(f"get_group_data_new_from_matrix called with custom_division={custom_division}, type={type(custom_division)}")
    print(f"number_of_states={number_of_states}, type={type(number_of_states)}")
    
    # Flatten the data matrix
    flattened_data = data_matrix.flatten()
    
    # Replace zeros with a small value
    flattened_data = np.where(flattened_data == 0, 0.001, flattened_data)
    
    # Sort the data
    #sorted_data = np.sort(flattened_data)
    sorted_data = flattened_data
    
    # Calculate the number of elements per group
    total_elements = len(sorted_data)
    elements_per_group = total_elements // int(number_of_states)
    
    # Initialize groups
    groups = []
    groups_stats = []
    
    if custom_division and (number_of_states == 4 or number_of_states == "4"):
        # Use the custom division values - these are absolute sizes, not indices
        division_points = [21080, 19880, 21072, 20912]
        # Total up all elements
        total_division_points = sum(division_points)
        
        print(f"Using custom division with points: {division_points}")
        print(f"Total custom division elements: {total_division_points}")
        print(f"Total data elements: {len(sorted_data)}")
        
        # Create groups based on custom division sizes
        start_idx = 0
        for i, size in enumerate(division_points):
            # Calculate what percentage of the data this division should use
            size_ratio = size / total_division_points
            # Calculate how many elements that corresponds to in our actual data
            actual_size = int(size_ratio * len(sorted_data))
            end_idx = min(start_idx + actual_size, len(sorted_data))  # Ensure we don't go beyond array bounds
            
            print(f"Group {i}: start_idx={start_idx}, end_idx={end_idx}, size={actual_size}")
            
            # Get the group data
            group_data = sorted_data[start_idx:end_idx]
            groups.append(group_data.tolist())
            
            # Calculate statistics
            min_val = np.min(group_data)
            max_val = np.max(group_data)
            avg_val = np.mean(group_data)
            std_val = np.std(group_data)
            groups_stats.append((min_val, max_val, avg_val, std_val))
            
            # Update start index for next group
            start_idx = end_idx
    else:
        # Create groups with equal number of elements
        for i in range(int(number_of_states)):
            start_idx = i * elements_per_group
            end_idx = (i + 1) * elements_per_group if i < int(number_of_states) - 1 else total_elements
            group_data = sorted_data[start_idx:end_idx]
            groups.append(group_data.tolist())
            
            # Calculate statistics
            min_val = np.min(group_data)
            max_val = np.max(group_data)
            avg_val = np.mean(group_data)
            std_val = np.std(group_data)
            groups_stats.append((min_val, max_val, avg_val, std_val))

    # Parse selected_groups or use default
    if selected_groups:
        try:
            if isinstance(selected_groups, str):
                selected_groups = [int(g) for g in selected_groups.split(',')]
            elif isinstance(selected_groups, list):
                selected_groups = [int(g) for g in selected_groups]
        except ValueError:
            selected_groups = list(range(int(number_of_states)))
    else:
        selected_groups = list(range(int(number_of_states)))
    
    return groups, groups_stats, selected_groups

def get_group_data_1124_2_from_matrix(target_ranges, data_matrix, selected_groups, pattern_file_array):
    """Modified version of get_group_data_1124_2 that works with a data matrix directly."""
    # Get groups data using existing function
    groups, groups_stats, selected_groups = get_group_data_from_matrix(data_matrix, selected_groups, pattern_file_array)
    
    # Calculate BER using target ranges
    table_miao_ber = calculate_ber_with_target_ranges(groups, target_ranges)
    
    return groups, groups_stats, selected_groups, table_miao_ber

def get_group_data_latest_from_matrix(target_ranges, data_matrix, selected_groups, number_of_states, custom_division=False):
    """Modified version of get_group_data_latest that works with a data matrix directly."""
    # Get groups data using existing function
    groups, groups_stats, selected_groups = get_group_data_new_from_matrix(data_matrix, selected_groups, number_of_states, custom_division)
    
    # Calculate BER using target ranges
    table_miao_ber = calculate_ber_with_target_ranges(groups, target_ranges)
    
    return groups, groups_stats, selected_groups, table_miao_ber