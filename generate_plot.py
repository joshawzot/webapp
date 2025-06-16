from tools_for_plots import *
import io
import base64
import pandas as pd
import re
from conductance_calculator import convert_table_to_conductance, convert_table_to_linear
import matplotlib.pyplot as plt

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
        if level * 2 + 1 < len(target_ranges):
            lower_bound, upper_bound = target_ranges[level * 2], target_ranges[level * 2 + 1]
            level_data = np.array(groups[level])
            if len(level_data) > 0:
                # Data is already filtered in the group processing, but double-check for NaN
                valid_data = level_data[~np.isnan(level_data)] if len(level_data) > 0 else np.array([])
                if len(valid_data) > 0:
                    out_of_range_data = valid_data[
                        (valid_data < lower_bound) | (valid_data > upper_bound)
                    ]
                    ber_values[f"State{level}"] = round(len(out_of_range_data) / len(valid_data) * 1e6)
                else:
                    ber_values[f"State{level}"] = 0
            else:
                ber_values[f"State{level}"] = 0
        else:
            ber_values[f"State{level}"] = 0

    # Transition BER (between consecutive levels)
    for level in range(num_levels - 1):
        level1_data = np.array(groups[level]) if len(groups[level]) > 0 else np.array([])
        level2_data = np.array(groups[level + 1]) if len(groups[level + 1]) > 0 else np.array([])
        
        # Data is already filtered, but double-check for NaN
        level1_data = level1_data[~np.isnan(level1_data)] if len(level1_data) > 0 else np.array([])
        level2_data = level2_data[~np.isnan(level2_data)] if len(level2_data) > 0 else np.array([])
        
        if len(level1_data) > 0 and len(level2_data) > 0:
            combined_data = np.concatenate([level1_data, level2_data])
            if (level * 2 + 3) < len(target_ranges):
                lower_bound1, upper_bound1 = target_ranges[level * 2], target_ranges[level * 2 + 1]
                lower_bound2, upper_bound2 = target_ranges[(level + 1) * 2], target_ranges[(level + 1) * 2 + 1]
                
                # Calculate transition BER
                out_of_range_data = combined_data[
                    (combined_data < min(lower_bound1, lower_bound2)) | 
                    (combined_data > max(upper_bound1, upper_bound2))
                ]
                ber_values[f"State{level}to{level + 1}"] = round(len(out_of_range_data) / len(combined_data) * 1e6)
            else:
                ber_values[f"State{level}to{level + 1}"] = 0
        else:
            ber_values[f"State{level}to{level + 1}"] = 0

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
    
    # Get BER range limits
    ber_lower_limit = form_data.get('ber_lower_limit')
    ber_upper_limit = form_data.get('ber_upper_limit')
    print("BER filter range:", ber_lower_limit, "to", ber_upper_limit)
    
    # Get analysis type for column-by-column analysis
    analysis_type = form_data.get('analysis_type', 'default')
    print("analysis_type:", analysis_type)
    
    # Initialize sigma_distances and num_states at the start
    sigma_distances = {}
    num_states = 0
    color_group_keywords = form_data.get('color_group_keywords', [])
    
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
            "1296x64_rowbar_4states": "State_pattern_files/1296x64_rowbar_4states.npy",
            "3x4_4states_debug": "State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "State_pattern_files/248x248_16states.npy",
            "248x1_1state": "State_pattern_files/248x1_1state.npy",
            "82944x78_ecc_fuxi": "State_pattern_files/82944x78_ecc_fuxi.npy"
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
    #colors = get_colors(len(table_names))
    avg_values = []
    std_values = []
    miao_ber = []
    sub_array_size = []

    # Compute the global min and max values among all data matrices
    data_matrices = []
    bitmap_mask_to_save = None  # Will store the bitmap mask if we need to generate one
    
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
        
        # Check if we need to apply an existing bitmap mask
        apply_bitmap_mask = form_data.get('apply_bitmap_mask', '').strip()
        if apply_bitmap_mask:
            print(f"Applying bitmap mask '{apply_bitmap_mask}' to table {table_name}")
            from route_handlers import load_bitmap_mask, validate_mask_dimensions
            
            mask_array, mask_dimensions = load_bitmap_mask(database_name, apply_bitmap_mask)
            if mask_array is not None:
                # Validate dimensions
                is_valid, validation_message = validate_mask_dimensions(mask_array, data_matrix)
                if is_valid:
                    print(f"Bitmap mask validation passed: {validation_message}")
                    # Apply the mask: set filtered coordinates to NaN
                    data_matrix[mask_array == 0] = np.nan
                    valid_count = np.sum(mask_array == 1)
                    filtered_count = np.sum(mask_array == 0)
                    print(f"Bitmap mask applied to {table_name}: {valid_count} valid points, {filtered_count} filtered out")
                else:
                    print(f"Bitmap mask validation failed for {table_name}: {validation_message}")
                    print("Skipping bitmap mask application")
            else:
                print(f"Could not load bitmap mask '{apply_bitmap_mask}' for table {table_name}")
        
        # Initialize combined mask for bitmap generation
        combined_filter_mask = np.ones(data_matrix.shape, dtype=bool)
        
        # Apply data range filtering if specified
        data_min_value = form_data.get('data_min_value')
        data_max_value = form_data.get('data_max_value')
        if data_min_value is not None or data_max_value is not None:
            print(f"Applying data range filter for table {table_name}: min={data_min_value}, max={data_max_value}")
            original_shape = data_matrix.shape
            original_count = data_matrix.size
            
            # Create a mask for values within the specified range
            mask = np.ones(data_matrix.shape, dtype=bool)
            if data_min_value is not None:
                mask &= (data_matrix >= data_min_value)
            if data_max_value is not None:
                mask &= (data_matrix <= data_max_value)
            
            # Update combined mask
            combined_filter_mask &= mask
            
            # Replace values outside the range with NaN
            filtered_data_matrix = data_matrix.copy()
            filtered_data_matrix[~mask] = np.nan
            
            # Count valid data points after filtering
            valid_count = np.sum(~np.isnan(filtered_data_matrix))
            filtered_count = original_count - valid_count
            
            print(f"Data filtering for {table_name}: {original_count} total points, {valid_count} valid points, {filtered_count} filtered out")
            data_matrix = filtered_data_matrix
            
        # Apply negative value filtering if specified
        filter_negative_values = form_data.get('filter_negative_values', False)
        if filter_negative_values:
            print(f"Applying negative value filter for table {table_name}")
            original_shape = data_matrix.shape
            original_count = np.sum(~np.isnan(data_matrix))  # Count non-NaN values before filtering
            
            # Create a mask for non-negative values (>= 0)
            negative_mask = (data_matrix >= 0)
            
            # Update combined mask
            combined_filter_mask &= negative_mask
            
            # Replace negative values with NaN
            filtered_data_matrix = data_matrix.copy()
            filtered_data_matrix[data_matrix < 0] = np.nan
            
            # Count valid data points after filtering
            valid_count = np.sum(~np.isnan(filtered_data_matrix))
            filtered_count = original_count - valid_count
            
            print(f"Negative value filtering for {table_name}: {original_count} non-NaN points before, {valid_count} valid points after, {filtered_count} negative values filtered out")
            data_matrix = filtered_data_matrix
        
        # Store bitmap mask for generation (use the first table's mask)
        generate_bitmap_mask = form_data.get('generate_bitmap_mask', False)
        if generate_bitmap_mask and bitmap_mask_to_save is None:
            # Convert boolean mask to integer (1 for valid, 0 for filtered)
            bitmap_mask_to_save = combined_filter_mask.astype(int)
            print(f"Bitmap mask prepared for generation from table {table_name}")
            
        data_matrices.append((table_name, data_matrix))
    
    # Ensure all data matrices are converted to float
    data_matrices = [(label, data_matrix.astype(float)) for label, data_matrix in data_matrices]

    print("min")
    global_min = min(np.min(data_matrix.astype(float)) for _, data_matrix in data_matrices)
    global_max = max(np.max(data_matrix.astype(float)) for _, data_matrix in data_matrices)
    g_range = (global_min, global_max)
    print("min")

    # Check if column-by-column analysis is enabled for 82944x78_ecc_fuxi
    column_analysis_enabled = (form_data['state_pattern_type'] == 'predefined' and 
                              form_data.get('state_pattern') == '82944x78_ecc_fuxi' and 
                              analysis_type == 'column_by_column')
    
    if column_analysis_enabled:
        print("Column-by-column analysis enabled for 82944x78_ecc_fuxi")
        # Handle column-by-column analysis
        return generate_column_by_column_analysis(table_names, database_name, form_data, data_matrices, 
                                                pattern_file_array, target_ranges, target_range_flag, 
                                                selected_groups, target_x_diff)
    
    # Process each table to extract groups and statistics (original logic)
    table_ber_data = {}  # Store BER data for each table
    table_indices = {}   # Map table names to their indices in the array
    all_ber_results = []  # Store all BER results for filtering

    for i, table_name in enumerate(table_names):
        # Use the already processed data matrix
        data_matrix = data_matrices[i][1]
        
        if target_range_flag == 0:
            if form_data['state_pattern_type'] == '1D':
                # Modify to use the data matrix directly
                groups, stats, selected_groups = get_group_data_new_from_matrix(
                    data_matrix, selected_groups, number_of_states, custom_division, form_data.get('custom_division_values', []))
            elif form_data['state_pattern_type'] == 'predefined':
                # Modify to use the data matrix directly
                groups, stats, selected_groups = get_group_data_from_matrix(
                    data_matrix, selected_groups, pattern_file_array)
        elif target_range_flag == 1:
            if form_data['state_pattern_type'] == '1D':
                # Modify to use the data matrix directly
                groups, stats, selected_groups, table_miao_ber = get_group_data_latest_from_matrix(
                    target_ranges, data_matrix, selected_groups, number_of_states, custom_division, form_data.get('custom_division_values', []))
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
        table_indices[table_name] = i

    # Generate colors based on keywords if provided
    if color_group_keywords:
        print(f"Using color group keywords: {color_group_keywords}")
        num_distinct_keyword_groups = len(color_group_keywords)
        num_colors_needed = num_distinct_keyword_groups + 1  # One for each keyword, one for 'other'
        
        cmap_name = 'tab20'  # tab20 has 20 distinct colors
        try:
            colormap = plt.colormaps[cmap_name]
        except AttributeError: # Older Matplotlib might not have plt.colormaps
            import matplotlib.cm as cm
            colormap = cm.get_cmap(cmap_name)

        distinct_colors_list = []
        if hasattr(colormap, 'colors'): # Qualitative colormap
            distinct_colors_list = list(colormap.colors)
        else: # Sequential/Diverging colormap, sample it
            for i in range(num_colors_needed):
                distinct_colors_list.append(colormap(i / max(1, num_colors_needed - 1)))

        if num_colors_needed > len(distinct_colors_list):
            print(f"Warning: Need {num_colors_needed} distinct colors, but 'cmap_name' provides {len(distinct_colors_list)}. Colors will be repeated.")
            original_cmap_colors = list(distinct_colors_list) # Use the fetched/generated list
            distinct_colors_list = [original_cmap_colors[i % len(original_cmap_colors)] for i in range(num_colors_needed)]

        keyword_colors_vals = distinct_colors_list[:num_distinct_keyword_groups]
        # Ensure other_color_val index is within bounds
        other_color_idx = num_distinct_keyword_groups % len(distinct_colors_list)
        other_color_val = distinct_colors_list[other_color_idx]

        keyword_to_color_map = {}
        for i, keyword in enumerate(color_group_keywords):
             # Ensure keyword_colors_vals index is within bounds if num_distinct_keyword_groups > len(distinct_colors_list)
            color_idx = i % len(keyword_colors_vals)
            keyword_to_color_map[keyword] = keyword_colors_vals[color_idx]
        
        colors = []
        for t_name in table_names:
            assigned_color = None
            for keyword in color_group_keywords:
                if keyword in t_name:  # Simple substring check
                    assigned_color = keyword_to_color_map[keyword]
                    break 
            if assigned_color is None:
                colors.append(other_color_val)
            else:
                colors.append(assigned_color)
    else:
        print("No color group keywords provided, using default colors.")
        colors = get_colors(len(table_names)) # from tools_for_plots.py

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

    # Calculate BER values using transformed CDF for each table
    print("Calculating BER values using transformed CDF...")
    temp_plot_data_sigma, temp_plot_data_cdf, temp_plot_data_interpolated_cdf, ber_results, sigma_intersections = plot_transformed_cdf_2(
        group_data, table_names, selected_groups, colors, target_x_diff, figsize=(15, 10), num_interp_points=form_data.get('num_interp_points', 500)
    )

    # Calculate max BER per table
    max_ber_per_table = {}
    for entry in ber_results:
        table_name = entry[0]
        ppm_ber = entry[4]  # ppm is at index 4 in the result tuple
        
        if ppm_ber is not None:
            # Update max BER for table
            if table_name not in max_ber_per_table or ppm_ber > max_ber_per_table[table_name]:
                max_ber_per_table[table_name] = ppm_ber

    # Sort table names by BER (low to high) for later use
    all_sorted_table_names = sorted(max_ber_per_table, key=max_ber_per_table.get, reverse=False)

    # Apply BER range filtering if limits are provided
    filtered_table_names = table_names[:]  # Start with all tables
    if ber_lower_limit is not None or ber_upper_limit is not None:
        filtered_table_names = []
        for table_name in table_names:
            # Get the max BER for this table
            table_ber = max_ber_per_table.get(table_name, 0)
            # Check if it's within the specified range
            if ((ber_lower_limit is None or table_ber >= ber_lower_limit) and 
                (ber_upper_limit is None or table_ber <= ber_upper_limit)):
                filtered_table_names.append(table_name)
        
        print(f"Filtered tables from {len(table_names)} to {len(filtered_table_names)} based on BER range")
        
        if len(filtered_table_names) == 0:
            print("Warning: No tables match the BER filter criteria!")
            # Return empty results to indicate no tables match
            return ([], [], None, None, None, None, [], [], [], None, None, {}, 0, [], {}, [])
    # Apply Top IOs filtering if ber_display_option is 'top_ios' and top_ios_count is specified
    elif form_data.get('ber_display_option') == 'top_ios':
        top_ios_count = form_data.get('top_ios_count')
        # Convert to integer if it's a non-empty string
        if top_ios_count and str(top_ios_count).strip():
            try:
                top_ios_count = int(top_ios_count)
                if top_ios_count > 0 and top_ios_count < len(all_sorted_table_names):
                    # Filter tables to only include the top N tables with best BER
                    filtered_table_names = all_sorted_table_names[:top_ios_count]
                    print(f"Filtered tables from {len(table_names)} to {len(filtered_table_names)} based on top BER count")
            except (ValueError, TypeError):
                pass  # If conversion fails, keep all tables (default behavior)

    # Create filtered versions of all data structures
    filtered_indices = [table_indices[name] for name in filtered_table_names]
    filtered_group_data = [group_data[i] for i in filtered_indices]
    filtered_avg_values = [avg_values[i] for i in filtered_indices]
    filtered_std_values = [std_values[i] for i in filtered_indices]
    filtered_colors = [colors[i] for i in filtered_indices]
    
    if target_range_flag == 1:
        filtered_miao_ber = {name: miao_ber[name] for name in filtered_table_names if name in miao_ber}
    
    filtered_data_matrices = [(name, matrix) for name, matrix in data_matrices if name in filtered_table_names]
    
    # Create a filtered version of sigma_distances if it exists
    filtered_sigma_distances = {}
    if sigma_distances:
        filtered_sigma_distances = {name: sigma_distances[name] for name in filtered_table_names if name in sigma_distances}
    
    # Create a filtered version of sigma_intersections
    filtered_sigma_intersections = {}
    for table_name in filtered_table_names:
        if table_name in sigma_intersections:
            filtered_sigma_intersections[table_name] = sigma_intersections[table_name]

    # Plot the color maps using the shared color scale if color_map_flag is True
    if color_map_flag:
        for table_name, data_matrix in filtered_data_matrices:
            if state_pattern in ("1296x64_rowbar_4states", "1296x64_Adrien_random_4states", "1296x64_1state"):
                encoded_plots.append(plot_colormap_magnified(
                    data_matrix, title=f"Colormap for {table_name}", g_range=g_range))
            else:
                encoded_plots.append(plot_colormap(
                    data_matrix, title=f"Colormap for {table_name}", g_range=g_range))

    # Generate plots for filtered tables
    encoded_plots.append(plot_boxplot(filtered_group_data, filtered_table_names))
    
    # Add data points table right after boxplot
    encoded_plots.append(plot_data_points_table(filtered_group_data, filtered_table_names, selected_groups))
    
    encoded_plots.append(plot_average_values_table(filtered_avg_values, filtered_table_names, selected_groups))
    encoded_plots.append(plot_std_values_table(filtered_std_values, filtered_table_names, selected_groups))

    # Get num_interp_points from form_data or use default
    num_interp_points = form_data.get('num_interp_points', 500)
    if isinstance(num_interp_points, str) and num_interp_points.strip():
        try:
            num_interp_points = int(num_interp_points)
            # Remove constraints to allow any value
        except ValueError:
            num_interp_points = 500  # Default if conversion fails
    elif not isinstance(num_interp_points, int):
        num_interp_points = 500  # Default if not an integer

    # Re-calculate plot data with filtered tables
    plot_data_sigma, plot_data_cdf, plot_data_interpolated_cdf, filtered_ber_results, filtered_sigma_intersections = plot_transformed_cdf_2(
        filtered_group_data, filtered_table_names, selected_groups, filtered_colors, target_x_diff, 
        figsize=(15, 10), num_interp_points=num_interp_points
    )
    
    encoded_plots.append(plot_data_sigma)
    encoded_plots.append(plot_data_cdf)
    encoded_plots.append(plot_data_interpolated_cdf)

    # Create a table for sigma intersections
    sigma_points = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    sigma_table = {}
    for table_name in filtered_table_names:
        if table_name in filtered_sigma_intersections:
            sigma_table[table_name] = filtered_sigma_intersections[table_name]

    if target_range_flag == 1:
        print("filtered_miao_ber:", filtered_miao_ber)
        encoded_plots.append(plot_miao(filtered_miao_ber))
    
    # Only perform outlier analysis if the flag is enabled and state_pattern_type is predefined
    if outlier_analysis_flag and form_data['state_pattern_type'] == 'predefined':
        try:
            # Process each filtered table's data for outliers
            for table_idx, table_name in enumerate(filtered_table_names):
                try:
                    # Get the data matrix for this table
                    data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
                    if not isinstance(data_matrix_size, tuple) or len(data_matrix_size) != 2:
                        print(f"Warning: Invalid data_matrix_size for table {table_name}")
                        continue
                        
                    rows, cols = data_matrix_size
                    print(f"Table {table_name} dimensions: {rows}x{cols}")
                    
                    # Get the last group's data
                    if len(filtered_group_data) > 0 and len(filtered_group_data[table_idx]) > 0:
                        last_group_idx = len(selected_groups) - 1 if selected_groups else 0
                        if last_group_idx >= len(filtered_group_data[table_idx]):
                            print(f"Warning: last_group_idx {last_group_idx} exceeds group_data length")
                            continue
                            
                        last_group = np.array(filtered_group_data[table_idx][last_group_idx], dtype=float)
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
                    first_table_name = filtered_table_names[0]
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
         sorted_table_names) = plot_ber_tables(filtered_ber_results, target_x_diff, num_interp_points)

        # Since we now have a combined image, append it to the plots
        encoded_plots.append(ppm_image)

        # Store original table names for the top IOs display
        original_sorted_table_names = sorted_table_names

        # Add the following code to create best_N and best_N_with_io based on user input
        # Get the top_ios_count from form_data or default to None
        top_ios_count = form_data.get('top_ios_count')
        # Convert to integer if it's a non-empty string
        if top_ios_count and str(top_ios_count).strip():
            try:
                top_ios_count = int(top_ios_count)
            except (ValueError, TypeError):
                top_ios_count = None
        else:
            top_ios_count = None
            
        # Ensure it doesn't exceed available tables
        if original_sorted_table_names:
            if top_ios_count and top_ios_count > 0:
                max_possible = len(original_sorted_table_names)
                top_ios_count = min(top_ios_count, max_possible)
                best_top_n = original_sorted_table_names[:top_ios_count]
            else:
                # If no value or zero, display all tables
                best_top_n = original_sorted_table_names
                
            best_top_n_with_io = ['io' + str(extract_number_from_table_name(name)) for name in best_top_n]
        else:
            best_top_n = []
            best_top_n_with_io = []

        # Save bitmap mask if requested
        generate_bitmap_mask = form_data.get('generate_bitmap_mask', False)
        bitmap_mask_name = form_data.get('bitmap_mask_name', '').strip()
        if generate_bitmap_mask and bitmap_mask_name and bitmap_mask_to_save is not None:
            print(f"Saving bitmap mask '{bitmap_mask_name}' for database '{database_name}'")
            from route_handlers import save_bitmap_mask
            success = save_bitmap_mask(database_name, bitmap_mask_name, bitmap_mask_to_save)
            if success:
                print(f"Bitmap mask '{bitmap_mask_name}' saved successfully")
            else:
                print(f"Failed to save bitmap mask '{bitmap_mask_name}'")

        # Return the plots and sorted table names
        return (encoded_plots,
                original_sorted_table_names,  # Use original table names
                None,  # Placeholder for sorted_table_names_100ppm (removed)
                None,  # Placeholder for sorted_table_names_200ppm (removed)
                None,  # Placeholder for sorted_table_names_500ppm (removed)
                None,  # Placeholder for sorted_table_names_1000ppm (removed)
                best_top_n,
                best_top_n_with_io,
                outlier_coordinates if outlier_analysis_flag else [],  # Only return outlier coordinates if flag is True
                correlation_analysis if outlier_analysis_flag else None,  # Only return correlation analysis if flag is True
                cluster_map if outlier_analysis_flag else None,
                filtered_sigma_distances,
                num_states,
                filtered_table_names,
                sigma_table,  # Add sigma intersections table
                sigma_points)  # Add sigma points
    else:
        sorted_table_names = None  # Handle the case where there is only one selected group

    # Handle the case where there is only one selected group
    if sorted_table_names is None:
        sorted_table_names = []
        best_top_n = []
        best_top_n_with_io = []
    else:
        # Store original table names
        original_sorted_table_names = sorted_table_names
        
        # Get the top_ios_count from form_data or default to None
        top_ios_count = form_data.get('top_ios_count')
        # Convert to integer if it's a non-empty string
        if top_ios_count and str(top_ios_count).strip():
            try:
                top_ios_count = int(top_ios_count)
            except (ValueError, TypeError):
                top_ios_count = None
        else:
            top_ios_count = None
            
        # Ensure it doesn't exceed available tables
        if original_sorted_table_names:
            if top_ios_count and top_ios_count > 0:
                max_possible = len(original_sorted_table_names)
                top_ios_count = min(top_ios_count, max_possible)
                best_top_n = original_sorted_table_names[:top_ios_count]
            else:
                # If no value or zero, display all tables
                best_top_n = original_sorted_table_names
                
            best_top_n_with_io = ['io' + str(extract_number_from_table_name(name)) for name in best_top_n]
        else:
            best_top_n = []
            best_top_n_with_io = []
    
    # Save bitmap mask if requested
    generate_bitmap_mask = form_data.get('generate_bitmap_mask', False)
    bitmap_mask_name = form_data.get('bitmap_mask_name', '').strip()
    if generate_bitmap_mask and bitmap_mask_name and bitmap_mask_to_save is not None:
        print(f"Saving bitmap mask '{bitmap_mask_name}' for database '{database_name}'")
        from route_handlers import save_bitmap_mask
        success = save_bitmap_mask(database_name, bitmap_mask_name, bitmap_mask_to_save)
        if success:
            print(f"Bitmap mask '{bitmap_mask_name}' saved successfully")
        else:
            print(f"Failed to save bitmap mask '{bitmap_mask_name}'")

    return (encoded_plots,
            sorted_table_names,
            None,
            None,
            None,
            None,
            best_top_n,
            best_top_n_with_io,
            outlier_coordinates if outlier_analysis_flag else [],
            correlation_analysis if outlier_analysis_flag else None,
            cluster_map if outlier_analysis_flag else None,
            filtered_sigma_distances,
            num_states,
            filtered_table_names,
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
        
        # Filter out NaN values (from data range filtering)
        values = [v for v in values if not np.isnan(v)]
        print(f"Group {group_idx}: {len(values)} valid values after filtering NaN")
        
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

def get_group_data_new_from_matrix(data_matrix, selected_groups, number_of_states, custom_division=False, custom_division_values=None):
    """Modified version of get_group_data_new that works with a data matrix directly."""
    # Debug prints to diagnose custom_division usage
    print(f"get_group_data_new_from_matrix called with custom_division={custom_division}, type={type(custom_division)}")
    print(f"number_of_states={number_of_states}, type={type(number_of_states)}")
    print(f"custom_division_values={custom_division_values}")
    
    # Flatten the data matrix
    flattened_data = data_matrix.flatten()
    
    # For 1D patterns, we need to divide into states FIRST, then filter within each state
    # This preserves the state boundaries even when data is filtered
    
    # Replace zeros with a small value BEFORE dividing into states
    flattened_data = np.where(flattened_data == 0, 0.001, flattened_data)
    
    # Sort the data (but keep track of original positions for state division)
    # For 1D patterns, we typically don't sort - we divide sequentially
    # sorted_data = np.sort(flattened_data)
    sorted_data = flattened_data
    
    # Calculate the number of elements per group based on ORIGINAL data size
    total_elements = len(sorted_data)
    elements_per_group = total_elements // int(number_of_states)
    
    # Initialize groups
    groups = []
    groups_stats = []
    
    if custom_division and (number_of_states == 4 or number_of_states == "4") and custom_division_values:
        # Use the custom division values from form_data
        division_points = custom_division_values
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
            
            # Get the group data (including NaN values)
            group_data = sorted_data[start_idx:end_idx]
            
            # NOW filter out NaN values within this specific group
            valid_mask = ~np.isnan(group_data)
            filtered_group_data = group_data[valid_mask]
            
            print(f"Group {i}: {len(group_data)} total points, {len(filtered_group_data)} valid points after filtering")
            
            # Store the filtered group data
            groups.append(filtered_group_data.tolist())
            
            # Calculate statistics on filtered data
            if len(filtered_group_data) > 0:
                min_val = np.min(filtered_group_data)
                max_val = np.max(filtered_group_data)
                avg_val = np.mean(filtered_group_data)
                std_val = np.std(filtered_group_data)
            else:
                # Handle case where all data in group was filtered out
                min_val = max_val = avg_val = std_val = 0.0
            
            groups_stats.append((min_val, max_val, avg_val, std_val))
            
            # Update start index for next group
            start_idx = end_idx
    else:
        # Create groups with equal number of elements based on ORIGINAL data
        for i in range(int(number_of_states)):
            start_idx = i * elements_per_group
            end_idx = (i + 1) * elements_per_group if i < int(number_of_states) - 1 else total_elements
            
            # Get the group data (including NaN values)
            group_data = sorted_data[start_idx:end_idx]
            
            # NOW filter out NaN values within this specific group
            valid_mask = ~np.isnan(group_data)
            filtered_group_data = group_data[valid_mask]
            
            print(f"Group {i}: {len(group_data)} total points, {len(filtered_group_data)} valid points after filtering")
            
            # Store the filtered group data
            groups.append(filtered_group_data.tolist())
            
            # Calculate statistics on filtered data
            if len(filtered_group_data) > 0:
                min_val = np.min(filtered_group_data)
                max_val = np.max(filtered_group_data)
                avg_val = np.mean(filtered_group_data)
                std_val = np.std(filtered_group_data)
            else:
                # Handle case where all data in group was filtered out
                min_val = max_val = avg_val = std_val = 0.0
            
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

def calculate_ber_with_target_ranges(groups, target_ranges):
    """Calculate BER values for different levels and transitions using target ranges."""
    num_levels = len(groups)
    ber_values = {}

    # Level n BER
    for level in range(num_levels):
        if level * 2 + 1 < len(target_ranges):
            lower_bound, upper_bound = target_ranges[level * 2], target_ranges[level * 2 + 1]
            level_data = np.array(groups[level])
            if len(level_data) > 0:
                # Data is already filtered in the group processing, but double-check for NaN
                valid_data = level_data[~np.isnan(level_data)] if len(level_data) > 0 else np.array([])
                if len(valid_data) > 0:
                    out_of_range_data = valid_data[
                        (valid_data < lower_bound) | (valid_data > upper_bound)
                    ]
                    ber_values[f"State{level}"] = round(len(out_of_range_data) / len(valid_data) * 1e6)
                else:
                    ber_values[f"State{level}"] = 0
            else:
                ber_values[f"State{level}"] = 0
        else:
            ber_values[f"State{level}"] = 0

    # Transition BER (between consecutive levels)
    for level in range(num_levels - 1):
        level1_data = np.array(groups[level]) if len(groups[level]) > 0 else np.array([])
        level2_data = np.array(groups[level + 1]) if len(groups[level + 1]) > 0 else np.array([])
        
        # Data is already filtered, but double-check for NaN
        level1_data = level1_data[~np.isnan(level1_data)] if len(level1_data) > 0 else np.array([])
        level2_data = level2_data[~np.isnan(level2_data)] if len(level2_data) > 0 else np.array([])
        
        if len(level1_data) > 0 and len(level2_data) > 0:
            combined_data = np.concatenate([level1_data, level2_data])
            if (level * 2 + 3) < len(target_ranges):
                lower_bound1, upper_bound1 = target_ranges[level * 2], target_ranges[level * 2 + 1]
                lower_bound2, upper_bound2 = target_ranges[(level + 1) * 2], target_ranges[(level + 1) * 2 + 1]
                
                # Calculate transition BER
                out_of_range_data = combined_data[
                    (combined_data < min(lower_bound1, lower_bound2)) | 
                    (combined_data > max(upper_bound1, upper_bound2))
                ]
                ber_values[f"State{level}to{level + 1}"] = round(len(out_of_range_data) / len(combined_data) * 1e6)
            else:
                ber_values[f"State{level}to{level + 1}"] = 0
        else:
            ber_values[f"State{level}to{level + 1}"] = 0

    return ber_values

def get_group_data_1124_2_from_matrix(target_ranges, data_matrix, selected_groups, pattern_file_array):
    """Modified version of get_group_data_1124_2 that works with a data matrix directly."""
    # Get groups data using existing function
    groups, groups_stats, selected_groups = get_group_data_from_matrix(data_matrix, selected_groups, pattern_file_array)
    
    # Calculate BER using target ranges
    table_miao_ber = calculate_ber_with_target_ranges(groups, target_ranges)
    
    return groups, groups_stats, selected_groups, table_miao_ber

def get_group_data_latest_from_matrix(target_ranges, data_matrix, selected_groups, number_of_states, custom_division=False, custom_division_values=None):
    """Modified version of get_group_data_latest that works with a data matrix directly."""
    # Get groups data using existing function
    groups, groups_stats, selected_groups = get_group_data_new_from_matrix(data_matrix, selected_groups, number_of_states, custom_division, custom_division_values)
    
    # Calculate BER using target ranges
    table_miao_ber = calculate_ber_with_target_ranges(groups, target_ranges)
    
    return groups, groups_stats, selected_groups, table_miao_ber

def generate_column_by_column_analysis(table_names, database_name, form_data, data_matrices, 
                                    pattern_file_array, target_ranges, target_range_flag, 
                                    selected_groups, target_x_diff):
    """
    Handle column-by-column analysis for 82944x78_ecc_fuxi pattern.
    Each of the 78 columns is treated as a separate 82944x1 dataset.
    """
    print("Processing column-by-column analysis for 82944x78_ecc_fuxi")
    
    # Initialize lists to store all column data
    all_column_group_data = []
    all_column_avg_values = []
    all_column_std_values = []
    all_column_miao_ber = []
    all_column_names = []
    
    # Process each table
    for table_idx, table_name in enumerate(table_names):
        data_matrix = data_matrices[table_idx][1]
        
        # For 82944x78_ecc_fuxi, data_matrix should be (82944, 78)
        if data_matrix.shape != (82944, 78):
            print(f"Warning: Expected shape (82944, 78) for {table_name}, got {data_matrix.shape}")
            # Try to reshape if possible
            if data_matrix.size == 82944 * 78:
                data_matrix = data_matrix.reshape(82944, 78)
                print(f"Reshaped {table_name} to (82944, 78)")
            else:
                print(f"Cannot reshape {table_name} to (82944, 78), skipping")
                continue
        
        # Get selected columns from form data
        selected_columns = form_data.get('selected_columns', list(range(78)))
        print(f"Processing selected columns: {selected_columns}")
        
        # Process only the selected columns
        for col_idx in selected_columns:
            # Check if column index is valid for this data matrix
            if col_idx >= data_matrix.shape[1]:
                print(f"Warning: Column {col_idx} is out of bounds for {table_name} (has {data_matrix.shape[1]} columns)")
                continue
                
            column_data = data_matrix[:, col_idx]  # Extract column as 82944x1
            column_name = f"{table_name}_Col{col_idx:02d}"  # e.g., "table_name_Col00", "table_name_Col01"
            all_column_names.append(column_name)
            
            # Create a pattern for this single column (all same state for simplicity)
            # For column analysis, we'll use the original pattern but only for this column
            column_pattern = pattern_file_array[:, col_idx:col_idx+1]  # Extract corresponding pattern column
            
            print(f"Processing column {col_idx} of {table_name}: {column_name}")
            
            # Reshape column data to match pattern expectations (82944, 1)
            column_data_reshaped = column_data.reshape(-1, 1)
            
            if target_range_flag == 0:
                # Use the matrix-based function for this column
                groups, stats, selected_groups_col = get_group_data_from_matrix(
                    column_data_reshaped, selected_groups, column_pattern)
            elif target_range_flag == 1:
                # Use the matrix-based function with target ranges for this column
                groups, stats, selected_groups_col, table_miao_ber = get_group_data_1124_2_from_matrix(
                    target_ranges, column_data_reshaped, selected_groups, column_pattern)
                all_column_miao_ber.append(table_miao_ber)
            
            # Extract average and standard deviation values for each selected group
            column_avg_values = [stat[2] for stat in stats]  # Index 2 is average
            column_std_values = [stat[3] for stat in stats]  # Index 3 is standard deviation
            
            all_column_group_data.append(groups)
            all_column_avg_values.append(column_avg_values)
            all_column_std_values.append(column_std_values)
    
    print(f"Processed {len(all_column_names)} columns total")
    
    # Generate colors for all columns
    num_columns = len(all_column_names)
    column_colors = get_colors(num_columns)
    
    # Calculate sigma distances if target values are provided
    sigma_distances = {}
    num_states = 0
    if form_data.get('target_values'):
        target_values = form_data.get('target_values', [])
        sigma_distances = calculate_sigma_distances(all_column_group_data, target_values, all_column_names)
        num_states = len(target_values)
    
    # Initialize encoded plots list
    encoded_plots = []
    
    # Generate plots for all columns
    encoded_plots.append(plot_boxplot(all_column_group_data, all_column_names))
    
    # For column analysis, create a custom data points summary instead of the standard table
    # since we have too many columns (78) for the standard table format
    encoded_plots.append(plot_column_data_points_summary(all_column_group_data, all_column_names, selected_groups))
    
    # Create custom column-friendly versions of the average and std tables
    encoded_plots.append(plot_column_average_values_summary(all_column_avg_values, all_column_names, selected_groups))
    encoded_plots.append(plot_column_std_values_summary(all_column_std_values, all_column_names, selected_groups))
    
    # Get num_interp_points from form_data or use default
    num_interp_points = form_data.get('num_interp_points', 500)
    if isinstance(num_interp_points, str) and num_interp_points.strip():
        try:
            num_interp_points = int(num_interp_points)
        except ValueError:
            num_interp_points = 500
    elif not isinstance(num_interp_points, int):
        num_interp_points = 500
    
    # Generate CDF and sigma plots
    plot_data_sigma, plot_data_cdf, plot_data_interpolated_cdf, column_ber_results, column_sigma_intersections = plot_transformed_cdf_2(
        all_column_group_data, all_column_names, selected_groups, column_colors, target_x_diff, 
        figsize=(15, 10), num_interp_points=num_interp_points
    )
    
    encoded_plots.append(plot_data_sigma)
    encoded_plots.append(plot_data_cdf)
    encoded_plots.append(plot_data_interpolated_cdf)
    
    # Generate BER tables if there are multiple selected groups
    if len(selected_groups) != 1:
        # Generate plots for BER results
        (sigma_image,
         ppm_image,
         uS_image,
         additional_image,
         sorted_column_names) = plot_ber_tables(column_ber_results, target_x_diff, num_interp_points)
        
        encoded_plots.append(ppm_image)
        
        # Create best column lists
        best_top_n = sorted_column_names[:10] if len(sorted_column_names) >= 10 else sorted_column_names
        best_top_n_with_io = [f"Col{col_name.split('_Col')[1]}" if '_Col' in col_name else col_name for col_name in best_top_n]
    else:
        sorted_column_names = []
        best_top_n = []
        best_top_n_with_io = []
    
    # Generate miao BER plot if target ranges are provided
    if target_range_flag == 1:
        combined_miao_ber = {}
        for i, col_name in enumerate(all_column_names):
            if i < len(all_column_miao_ber):
                combined_miao_ber[col_name] = all_column_miao_ber[i]
        encoded_plots.append(plot_miao(combined_miao_ber))
    
    # Create sigma intersections table
    sigma_points = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    sigma_table = {}
    for col_name in all_column_names:
        if col_name in column_sigma_intersections:
            sigma_table[col_name] = column_sigma_intersections[col_name]
    
    print(f"Column-by-column analysis complete. Generated {len(encoded_plots)} plots for {len(all_column_names)} columns.")
    
    # Return the same structure as the original function
    return (encoded_plots,
            sorted_column_names,
            None, None, None, None,  # Placeholders for removed fields
            best_top_n,
            best_top_n_with_io,
            [],  # outlier_coordinates (empty for column analysis)
            None,  # correlation_analysis (None for column analysis)
            None,  # cluster_map (None for column analysis)
            sigma_distances,
            num_states,
            all_column_names,  # Use column names instead of table names
            sigma_table,
            sigma_points)

def plot_column_data_points_summary(all_column_group_data, all_column_names, selected_groups):
    """
    Create a summary table for column-by-column analysis showing data point counts.
    This is a more compact version that can handle many columns (78 in this case).
    """
    try:
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        import numpy as np
        
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Build summary statistics
        summary_data = []
        
        # Create header
        header = ["Column", "Total Points", "Points per State"]
        summary_data.append(header)
        
        # Process each column
        for i, (column_name, group_data) in enumerate(zip(all_column_names, all_column_group_data)):
            # Extract just the column identifier (e.g., "Col00" from "table_name_Col00")
            col_id = column_name.split('_Col')[-1] if '_Col' in column_name else str(i)
            
            # Calculate total points across all states
            total_points = sum(len(subgroup) for subgroup in group_data)
            
            # Create points per state string
            state_points = []
            for j, subgroup in enumerate(group_data):
                if j < len(selected_groups):
                    state_points.append(f"S{selected_groups[j]}:{len(subgroup)}")
            
            points_per_state_str = " | ".join(state_points)
            
            row = [f"Col{col_id}", str(total_points), points_per_state_str]
            summary_data.append(row)
        
        # Calculate column widths
        col_widths = [0.15, 0.15, 0.7]  # Column ID, Total Points, Points per State
        
        # Create the table
        table = ax.table(cellText=summary_data, loc='center', colWidths=col_widths, cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the header row
        for i in range(len(col_widths)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors
        for i in range(1, len(summary_data)):
            for j in range(len(col_widths)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        # Set title
        ax.set_title('Column-by-Column Data Points Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Add subtitle with explanation
        subtitle = f"Each column shows data point counts for {len(selected_groups)} selected states (S{selected_groups})"
        ax.text(0.5, 0.95, subtitle, ha='center', va='top', transform=ax.transAxes, 
                fontsize=12, style='italic')

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
        
    except Exception as e:
        print(f"Error in plot_column_data_points_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_column_average_values_summary(all_column_avg_values, all_column_names, selected_groups):
    """
    Create an average values summary table for column-by-column analysis.
    This version can handle many columns (78 in this case) without hitting matplotlib limits.
    """
    try:
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        import numpy as np
        
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Build summary data
        summary_data = []
        
        # Create header - dynamic based on selected groups
        header = ["Column"] + [f"State {group} Avg" for group in selected_groups] + ["Max Avg", "Overall Avg"]
        summary_data.append(header)
        
        # Debug info
        print(f"Debug: selected_groups = {selected_groups}")
        print(f"Debug: header = {header}")
        print(f"Debug: header length = {len(header)}")
        if all_column_avg_values:
            print(f"Debug: first column avg_values length = {len(all_column_avg_values[0])}")
            print(f"Debug: first column avg_values = {all_column_avg_values[0]}")
        
        # Process each column's average values
        for i, (column_name, avg_values) in enumerate(zip(all_column_names, all_column_avg_values)):
            # Extract just the column identifier
            col_id = column_name.split('_Col')[-1] if '_Col' in column_name else str(i)
            
            row = [f"Col{col_id}"]
            
            # Debug: check if lengths match
            if len(avg_values) != len(selected_groups):
                print(f"Warning: Column {col_id} has {len(avg_values)} values but {len(selected_groups)} selected groups")
                # Pad or truncate to match selected_groups length
                while len(avg_values) < len(selected_groups):
                    avg_values.append(0.0)
                avg_values = avg_values[:len(selected_groups)]
            
            # Add average values for each selected state
            for avg_val in avg_values:
                row.append(f"{avg_val:.2f}")
            
            # Calculate max and overall average for this column
            max_avg = max(avg_values) if avg_values else 0.0
            overall_avg = np.mean(avg_values) if avg_values else 0.0
            row.append(f"{max_avg:.2f}")
            row.append(f"{overall_avg:.2f}")
            
            # Debug: verify row length matches header
            if len(row) != len(header):
                print(f"Error: Row {i} has {len(row)} columns, header has {len(header)} columns")
                print(f"Header: {header}")
                print(f"Row: {row}")
                # Fix the row length
                while len(row) < len(header):
                    row.append("N/A")
                row = row[:len(header)]
            
            summary_data.append(row)
        
        # Add summary statistics at the bottom
        if len(all_column_avg_values) > 0:
            # Calculate column-wise statistics
            state_averages = []
            for state_idx in range(len(selected_groups)):
                state_values = [col_avgs[state_idx] for col_avgs in all_column_avg_values if state_idx < len(col_avgs)]
                if state_values:
                    state_averages.append(np.mean(state_values))
                else:
                    state_averages.append(0.0)
            
            # Add separator row
            separator_row = ["---"] * len(header)
            summary_data.append(separator_row)
            
            # Add overall averages row
            overall_row = ["Overall Avg"] + [f"{avg:.2f}" for avg in state_averages]
            if state_averages:
                overall_row.append(f"{max(state_averages):.2f}")
                overall_row.append(f"{np.mean(state_averages):.2f}")
            else:
                overall_row.extend(["N/A", "N/A"])
            summary_data.append(overall_row)
        
        # Calculate column widths dynamically
        num_cols = len(header)
        col_width = 0.9 / num_cols  # Leave some margin
        col_widths = [col_width] * num_cols
        
        # Create the table
        table = ax.table(cellText=summary_data, loc='center', colWidths=col_widths, cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        
        # Style the header row
        for i in range(num_cols):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style separator row if it exists
        if len(summary_data) > len(all_column_avg_values) + 1:
            separator_row_idx = len(summary_data) - 2
            for i in range(num_cols):
                table[(separator_row_idx, i)].set_facecolor('#d0d0d0')
        
        # Style summary row
        if len(summary_data) > len(all_column_avg_values) + 1:
            summary_row_idx = len(summary_data) - 1
            for i in range(num_cols):
                table[(summary_row_idx, i)].set_facecolor('#e0e0ff')
                table[(summary_row_idx, i)].set_text_props(weight='bold')
        
        # Style data rows with alternating colors
        for i in range(1, len(all_column_avg_values) + 1):
            for j in range(num_cols):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        # Set title
        ax.set_title('Column-by-Column Average Values Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Add subtitle
        subtitle = f"Average values for each column across {len(selected_groups)} selected states (S{selected_groups})"
        ax.text(0.5, 0.95, subtitle, ha='center', va='top', transform=ax.transAxes, 
                fontsize=12, style='italic')

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
        
    except Exception as e:
        print(f"Error in plot_column_average_values_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()

def plot_column_std_values_summary(all_column_std_values, all_column_names, selected_groups):
    """
    Create a standard deviation values summary table for column-by-column analysis.
    This version can handle many columns (78 in this case) without hitting matplotlib limits.
    """
    try:
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        import numpy as np
        
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Build summary data
        summary_data = []
        
        # Create header - dynamic based on selected groups
        header = ["Column"] + [f"State {group} StdDev" for group in selected_groups] + ["Max StdDev", "Overall StdDev"]
        summary_data.append(header)
        
        # Debug info
        print(f"Debug: selected_groups = {selected_groups}")
        print(f"Debug: header = {header}")
        print(f"Debug: header length = {len(header)}")
        if all_column_std_values:
            print(f"Debug: first column std_values length = {len(all_column_std_values[0])}")
            print(f"Debug: first column std_values = {all_column_std_values[0]}")
        
        # Process each column's std dev values
        for i, (column_name, std_values) in enumerate(zip(all_column_names, all_column_std_values)):
            # Extract just the column identifier
            col_id = column_name.split('_Col')[-1] if '_Col' in column_name else str(i)
            
            row = [f"Col{col_id}"]
            
            # Debug: check if lengths match
            if len(std_values) != len(selected_groups):
                print(f"Warning: Column {col_id} has {len(std_values)} values but {len(selected_groups)} selected groups")
                # Pad or truncate to match selected_groups length
                while len(std_values) < len(selected_groups):
                    std_values.append(0.0)
                std_values = std_values[:len(selected_groups)]
            
            # Add std dev values for each selected state
            for std_val in std_values:
                row.append(f"{std_val:.2f}")
            
            # Calculate max and overall std dev for this column
            max_std = max(std_values) if std_values else 0.0
            overall_std = np.mean(std_values) if std_values else 0.0
            row.append(f"{max_std:.2f}")
            row.append(f"{overall_std:.2f}")
            
            # Debug: verify row length matches header
            if len(row) != len(header):
                print(f"Error: Row {i} has {len(row)} columns, header has {len(header)} columns")
                print(f"Header: {header}")
                print(f"Row: {row}")
                # Fix the row length
                while len(row) < len(header):
                    row.append("N/A")
                row = row[:len(header)]
            
            summary_data.append(row)
        
        # Add summary statistics at the bottom
        if len(all_column_std_values) > 0:
            # Calculate column-wise statistics
            state_std_devs = []
            for state_idx in range(len(selected_groups)):
                state_values = [col_stds[state_idx] for col_stds in all_column_std_values if state_idx < len(col_stds)]
                if state_values:
                    state_std_devs.append(np.mean(state_values))
                else:
                    state_std_devs.append(0.0)
            
            # Add separator row
            separator_row = ["---"] * len(header)
            summary_data.append(separator_row)
            
            # Add overall std devs row
            overall_row = ["Overall StdDev"] + [f"{std:.2f}" for std in state_std_devs]
            if state_std_devs:
                overall_row.append(f"{max(state_std_devs):.2f}")
                overall_row.append(f"{np.mean(state_std_devs):.2f}")
            else:
                overall_row.extend(["N/A", "N/A"])
            summary_data.append(overall_row)
        
        # Calculate column widths dynamically
        num_cols = len(header)
        col_width = 0.9 / num_cols  # Leave some margin
        col_widths = [col_width] * num_cols
        
        # Create the table
        table = ax.table(cellText=summary_data, loc='center', colWidths=col_widths, cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        
        # Style the header row
        for i in range(num_cols):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style separator row if it exists
        if len(summary_data) > len(all_column_std_values) + 1:
            separator_row_idx = len(summary_data) - 2
            for i in range(num_cols):
                table[(separator_row_idx, i)].set_facecolor('#d0d0d0')
        
        # Style summary row
        if len(summary_data) > len(all_column_std_values) + 1:
            summary_row_idx = len(summary_data) - 1
            for i in range(num_cols):
                table[(summary_row_idx, i)].set_facecolor('#e0e0ff')
                table[(summary_row_idx, i)].set_text_props(weight='bold')
        
        # Style data rows with alternating colors
        for i in range(1, len(all_column_std_values) + 1):
            for j in range(num_cols):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        # Set title
        ax.set_title('Column-by-Column Standard Deviation Values Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Add subtitle
        subtitle = f"Standard deviation values for each column across {len(selected_groups)} selected states (S{selected_groups})"
        ax.text(0.5, 0.95, subtitle, ha='center', va='top', transform=ax.transAxes, 
                fontsize=12, style='italic')

        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
        
    except Exception as e:
        print(f"Error in plot_column_std_values_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        plt.close(fig)
        if 'buf' in locals():
            buf.close()