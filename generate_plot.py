from tools_for_plots import *
import io
import base64
import pandas as pd
import re
from conductance_calculator import convert_table_to_conductance, convert_table_to_linear
import matplotlib.pyplot as plt

def get_pattern_files():
    """
    Get the pattern files dictionary based on the configuration setting in run.py
    Returns dictionary with appropriate paths (absolute or relative)
    """
    try:
        # Import Flask app to access configuration
        from run import app
        use_absolute_paths = app.config.get('USE_ABSOLUTE_STATE_PATTERN_PATHS', True)
    except:
        # Fallback to True if import fails or config not found
        use_absolute_paths = True
    
    if use_absolute_paths:
        # Absolute paths dictionary
        return {
            "1296x64_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_rowbar_4states.npy",
            "2048x32_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/2048x32_rowbar_4states.npy",
            "2048x32_random": "/home/admin2/webapp_2/State_pattern_files/2048x32_random.npy",
            "3x4_4states_debug": "/home/admin2/webapp_2/State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "/home/admin2/webapp_2/State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "/home/admin2/webapp_2/State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "/home/admin2/webapp_2/State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "/home/admin2/webapp_2/State_pattern_files/248x248_16states.npy",
            "248x248_2states": "/home/admin2/webapp_2/State_pattern_files/248x248_2states.npy",
            "248x248_64states": "/home/admin2/webapp_2/State_pattern_files/248x248_64states.npy",
            "62x62_2states": "/home/admin2/webapp_2/State_pattern_files/62x62_2states.npy",
            "248x1_1state": "/home/admin2/webapp_2/State_pattern_files/248x1_1state.npy",
            "248x256_1state": "/home/admin2/webapp_2/State_pattern_files/248x256_1state.npy",
            "82944x78_ecc_fuxi": "/home/admin2/webapp_2/State_pattern_files/82944x78_ecc_fuxi.npy",
            "65536x78_ecc": "/home/admin2/webapp_2/State_pattern_files/65536x78_ecc.npy",
            "256x32_pr0": "/home/admin2/webapp_2/State_pattern_files/256x32_pr0.npy",
            "256x32_pr1": "/home/admin2/webapp_2/State_pattern_files/256x32_pr1.npy",
            "test_chin": "/home/admin2/webapp_2/State_pattern_files/ecc_new.npy",
            # ECC 2048x32 IO files
            "ecc_2048x32_IO0": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO0.npy",
            "ecc_2048x32_IO1": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO1.npy",
            "ecc_2048x32_IO2": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO2.npy",
            "ecc_2048x32_IO3": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO3.npy",
            "ecc_2048x32_IO4": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO4.npy",
            "ecc_2048x32_IO5": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO5.npy",
            "ecc_2048x32_IO6": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO6.npy",
            "ecc_2048x32_IO7": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO7.npy",
            "ecc_2048x32_IO8": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO8.npy",
            "ecc_2048x32_IO9": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO9.npy",
            "ecc_2048x32_IO10": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO10.npy",
            "ecc_2048x32_IO11": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO11.npy",
            "ecc_2048x32_IO12": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO12.npy",
            "ecc_2048x32_IO13": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO13.npy",
            "ecc_2048x32_IO14": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO14.npy",
            "ecc_2048x32_IO15": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO15.npy",
            "ecc_2048x32_IO16": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO16.npy",
            "ecc_2048x32_IO17": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO17.npy",
            "ecc_2048x32_IO18": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO18.npy",
            "ecc_2048x32_IO19": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO19.npy",
            "ecc_2048x32_IO20": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO20.npy",
            "ecc_2048x32_IO21": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO21.npy",
            "ecc_2048x32_IO22": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO22.npy",
            "ecc_2048x32_IO23": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO23.npy",
            "ecc_2048x32_IO24": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO24.npy",
            "ecc_2048x32_IO25": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO25.npy",
            "ecc_2048x32_IO26": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO26.npy",
            "ecc_2048x32_IO27": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO27.npy",
            "ecc_2048x32_IO28": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO28.npy",
            "ecc_2048x32_IO29": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO29.npy",
            "ecc_2048x32_IO30": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO30.npy",
            "ecc_2048x32_IO31": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO31.npy",
            "ecc_2048x32_IO32": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO32.npy",
            "ecc_2048x32_IO33": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO33.npy",
            "ecc_2048x32_IO34": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO34.npy",
            "ecc_2048x32_IO35": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO35.npy",
            "ecc_2048x32_IO36": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO36.npy",
            "ecc_2048x32_IO37": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO37.npy",
            "ecc_2048x32_IO38": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO38.npy",
            "ecc_2048x32_IO39": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO39.npy",
            "ecc_2048x32_IO40": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO40.npy",
            "ecc_2048x32_IO41": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO41.npy",
            "ecc_2048x32_IO42": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO42.npy",
            "ecc_2048x32_IO43": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO43.npy",
            "ecc_2048x32_IO44": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO44.npy",
            "ecc_2048x32_IO45": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO45.npy",
            "ecc_2048x32_IO46": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO46.npy",
            "ecc_2048x32_IO47": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO47.npy",
            "ecc_2048x32_IO48": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO48.npy",
            "ecc_2048x32_IO49": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO49.npy",
            "ecc_2048x32_IO50": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO50.npy",
            "ecc_2048x32_IO51": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO51.npy",
            "ecc_2048x32_IO52": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO52.npy",
            "ecc_2048x32_IO53": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO53.npy",
            "ecc_2048x32_IO54": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO54.npy",
            "ecc_2048x32_IO55": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO55.npy",
            "ecc_2048x32_IO56": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO56.npy",
            "ecc_2048x32_IO57": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO57.npy",
            "ecc_2048x32_IO58": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO58.npy",
            "ecc_2048x32_IO59": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO59.npy",
            "ecc_2048x32_IO60": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO60.npy",
            "ecc_2048x32_IO61": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO61.npy",
            "ecc_2048x32_IO62": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO62.npy",
            "ecc_2048x32_IO63": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO63.npy",
            "ecc_2048x32_IO64": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO64.npy",
            "ecc_2048x32_IO65": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO65.npy",
            "ecc_2048x32_IO66": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO66.npy",
            "ecc_2048x32_IO67": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO67.npy",
            "ecc_2048x32_IO68": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO68.npy",
            "ecc_2048x32_IO69": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO69.npy",
            "ecc_2048x32_IO70": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO70.npy",
            "ecc_2048x32_IO71": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO71.npy",
            "ecc_2048x32_IO72": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO72.npy",
            "ecc_2048x32_IO73": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO73.npy",
            "ecc_2048x32_IO74": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO74.npy",
            "ecc_2048x32_IO75": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO75.npy",
            "ecc_2048x32_IO76": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO76.npy",
            "ecc_2048x32_IO77": "/home/admin2/webapp_2/State_pattern_files/ecc_2048x32_IO77.npy",
        }
    else:
        # Relative paths dictionary
        return {
            "1296x64_rowbar_4states": "State_pattern_files/1296x64_rowbar_4states.npy",
            "2048x32_rowbar_4states": "State_pattern_files/2048x32_rowbar_4states.npy",
            "2048x32_random": "State_pattern_files/2048x32_random.npy",
            "3x4_4states_debug": "State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "State_pattern_files/248x248_16states.npy",
            "248x248_2states": "State_pattern_files/248x248_2states.npy",
            "248x248_64states": "State_pattern_files/248x248_64states.npy",
            "62x62_2states": "State_pattern_files/62x62_2states.npy",
            "248x1_1state": "State_pattern_files/248x1_1state.npy",
            "248x256_1state": "State_pattern_files/248x256_1state.npy",
            "82944x78_ecc_fuxi": "State_pattern_files/82944x78_ecc_fuxi.npy",
            "65536x78_ecc": "State_pattern_files/65536x78_ecc.npy",
            "256x32_pr0": "State_pattern_files/256x32_pr0.npy",
            "256x32_pr1": "State_pattern_files/256x32_pr1.npy",
            "test_chin": "State_pattern_files/ecc_new.npy",
            # ECC 2048x32 IO files
            "ecc_2048x32_IO0": "State_pattern_files/ecc_2048x32_IO0.npy",
            "ecc_2048x32_IO1": "State_pattern_files/ecc_2048x32_IO1.npy",
            "ecc_2048x32_IO2": "State_pattern_files/ecc_2048x32_IO2.npy",
            "ecc_2048x32_IO3": "State_pattern_files/ecc_2048x32_IO3.npy",
            "ecc_2048x32_IO4": "State_pattern_files/ecc_2048x32_IO4.npy",
            "ecc_2048x32_IO5": "State_pattern_files/ecc_2048x32_IO5.npy",
            "ecc_2048x32_IO6": "State_pattern_files/ecc_2048x32_IO6.npy",
            "ecc_2048x32_IO7": "State_pattern_files/ecc_2048x32_IO7.npy",
            "ecc_2048x32_IO8": "State_pattern_files/ecc_2048x32_IO8.npy",
            "ecc_2048x32_IO9": "State_pattern_files/ecc_2048x32_IO9.npy",
            "ecc_2048x32_IO10": "State_pattern_files/ecc_2048x32_IO10.npy",
            "ecc_2048x32_IO11": "State_pattern_files/ecc_2048x32_IO11.npy",
            "ecc_2048x32_IO12": "State_pattern_files/ecc_2048x32_IO12.npy",
            "ecc_2048x32_IO13": "State_pattern_files/ecc_2048x32_IO13.npy",
            "ecc_2048x32_IO14": "State_pattern_files/ecc_2048x32_IO14.npy",
            "ecc_2048x32_IO15": "State_pattern_files/ecc_2048x32_IO15.npy",
            "ecc_2048x32_IO16": "State_pattern_files/ecc_2048x32_IO16.npy",
            "ecc_2048x32_IO17": "State_pattern_files/ecc_2048x32_IO17.npy",
            "ecc_2048x32_IO18": "State_pattern_files/ecc_2048x32_IO18.npy",
            "ecc_2048x32_IO19": "State_pattern_files/ecc_2048x32_IO19.npy",
            "ecc_2048x32_IO20": "State_pattern_files/ecc_2048x32_IO20.npy",
            "ecc_2048x32_IO21": "State_pattern_files/ecc_2048x32_IO21.npy",
            "ecc_2048x32_IO22": "State_pattern_files/ecc_2048x32_IO22.npy",
            "ecc_2048x32_IO23": "State_pattern_files/ecc_2048x32_IO23.npy",
            "ecc_2048x32_IO24": "State_pattern_files/ecc_2048x32_IO24.npy",
            "ecc_2048x32_IO25": "State_pattern_files/ecc_2048x32_IO25.npy",
            "ecc_2048x32_IO26": "State_pattern_files/ecc_2048x32_IO26.npy",
            "ecc_2048x32_IO27": "State_pattern_files/ecc_2048x32_IO27.npy",
            "ecc_2048x32_IO28": "State_pattern_files/ecc_2048x32_IO28.npy",
            "ecc_2048x32_IO29": "State_pattern_files/ecc_2048x32_IO29.npy",
            "ecc_2048x32_IO30": "State_pattern_files/ecc_2048x32_IO30.npy",
            "ecc_2048x32_IO31": "State_pattern_files/ecc_2048x32_IO31.npy",
            "ecc_2048x32_IO32": "State_pattern_files/ecc_2048x32_IO32.npy",
            "ecc_2048x32_IO33": "State_pattern_files/ecc_2048x32_IO33.npy",
            "ecc_2048x32_IO34": "State_pattern_files/ecc_2048x32_IO34.npy",
            "ecc_2048x32_IO35": "State_pattern_files/ecc_2048x32_IO35.npy",
            "ecc_2048x32_IO36": "State_pattern_files/ecc_2048x32_IO36.npy",
            "ecc_2048x32_IO37": "State_pattern_files/ecc_2048x32_IO37.npy",
            "ecc_2048x32_IO38": "State_pattern_files/ecc_2048x32_IO38.npy",
            "ecc_2048x32_IO39": "State_pattern_files/ecc_2048x32_IO39.npy",
            "ecc_2048x32_IO40": "State_pattern_files/ecc_2048x32_IO40.npy",
            "ecc_2048x32_IO41": "State_pattern_files/ecc_2048x32_IO41.npy",
            "ecc_2048x32_IO42": "State_pattern_files/ecc_2048x32_IO42.npy",
            "ecc_2048x32_IO43": "State_pattern_files/ecc_2048x32_IO43.npy",
            "ecc_2048x32_IO44": "State_pattern_files/ecc_2048x32_IO44.npy",
            "ecc_2048x32_IO45": "State_pattern_files/ecc_2048x32_IO45.npy",
            "ecc_2048x32_IO46": "State_pattern_files/ecc_2048x32_IO46.npy",
            "ecc_2048x32_IO47": "State_pattern_files/ecc_2048x32_IO47.npy",
            "ecc_2048x32_IO48": "State_pattern_files/ecc_2048x32_IO48.npy",
            "ecc_2048x32_IO49": "State_pattern_files/ecc_2048x32_IO49.npy",
            "ecc_2048x32_IO50": "State_pattern_files/ecc_2048x32_IO50.npy",
            "ecc_2048x32_IO51": "State_pattern_files/ecc_2048x32_IO51.npy",
            "ecc_2048x32_IO52": "State_pattern_files/ecc_2048x32_IO52.npy",
            "ecc_2048x32_IO53": "State_pattern_files/ecc_2048x32_IO53.npy",
            "ecc_2048x32_IO54": "State_pattern_files/ecc_2048x32_IO54.npy",
            "ecc_2048x32_IO55": "State_pattern_files/ecc_2048x32_IO55.npy",
            "ecc_2048x32_IO56": "State_pattern_files/ecc_2048x32_IO56.npy",
            "ecc_2048x32_IO57": "State_pattern_files/ecc_2048x32_IO57.npy",
            "ecc_2048x32_IO58": "State_pattern_files/ecc_2048x32_IO58.npy",
            "ecc_2048x32_IO59": "State_pattern_files/ecc_2048x32_IO59.npy",
            "ecc_2048x32_IO60": "State_pattern_files/ecc_2048x32_IO60.npy",
            "ecc_2048x32_IO61": "State_pattern_files/ecc_2048x32_IO61.npy",
            "ecc_2048x32_IO62": "State_pattern_files/ecc_2048x32_IO62.npy",
            "ecc_2048x32_IO63": "State_pattern_files/ecc_2048x32_IO63.npy",
            "ecc_2048x32_IO64": "State_pattern_files/ecc_2048x32_IO64.npy",
            "ecc_2048x32_IO65": "State_pattern_files/ecc_2048x32_IO65.npy",
            "ecc_2048x32_IO66": "State_pattern_files/ecc_2048x32_IO66.npy",
            "ecc_2048x32_IO67": "State_pattern_files/ecc_2048x32_IO67.npy",
            "ecc_2048x32_IO68": "State_pattern_files/ecc_2048x32_IO68.npy",
            "ecc_2048x32_IO69": "State_pattern_files/ecc_2048x32_IO69.npy",
            "ecc_2048x32_IO70": "State_pattern_files/ecc_2048x32_IO70.npy",
            "ecc_2048x32_IO71": "State_pattern_files/ecc_2048x32_IO71.npy",
            "ecc_2048x32_IO72": "State_pattern_files/ecc_2048x32_IO72.npy",
            "ecc_2048x32_IO73": "State_pattern_files/ecc_2048x32_IO73.npy",
            "ecc_2048x32_IO74": "State_pattern_files/ecc_2048x32_IO74.npy",
            "ecc_2048x32_IO75": "State_pattern_files/ecc_2048x32_IO75.npy",
            "ecc_2048x32_IO76": "State_pattern_files/ecc_2048x32_IO76.npy",
            "ecc_2048x32_IO77": "State_pattern_files/ecc_2048x32_IO77.npy",
        }

def get_group_data_1124(table_name, selected_groups, database_name, pattern_file_array, exclude_ranges_type='', exclude_ranges=[]):
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
    print(f"🔍 DIMENSION CHECK: pattern_file_array.shape = {pattern_file_array.shape}")
    print(f"🔍 DIMENSION CHECK: data_np.shape = {data_np.shape}")
    
    if pattern_file_array.shape != data_np.shape:
        print(f"🚨 DIMENSION MISMATCH DETECTED!")
        print(f"   Pattern shape: {pattern_file_array.shape}")
        print(f"   Data shape: {data_np.shape}")
        raise ValueError(f"pattern_file_array shape {pattern_file_array.shape} must have the same shape as the data array {data_np.shape}.")
    else:
        print(f"✅ DIMENSION CHECK PASSED: shapes match {pattern_file_array.shape}")

    # Create exclude mask if exclude ranges are specified
    exclude_mask = create_exclude_mask(data_np.shape, exclude_ranges_type, exclude_ranges)
    if exclude_ranges_type and exclude_ranges:
        print(f"Applied exclude mask: {exclude_ranges_type} {exclude_ranges}")
        excluded_count = np.sum(~exclude_mask)
        total_count = exclude_mask.size
        print(f"Excluding {excluded_count} out of {total_count} data points ({excluded_count/total_count*100:.1f}%)")

    unique_groups = np.unique(pattern_file_array)
    group_indices = unique_groups.tolist()
    print("group_indices:", group_indices)  # e.g., [0, 1, 2, 3]

    for group_idx in group_indices:
        if group_idx in selected_groups:
            # Get the mask where pattern_file_array equals group_idx
            group_mask = np.equal(pattern_file_array, group_idx)  # Use np.equal instead of == for better array handling
            # Combine group mask with exclude mask
            combined_mask = group_mask & exclude_mask
            group_data = data_np[combined_mask]
            
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
    print(f"🔍 DIMENSION CHECK: pattern_file_array.shape = {pattern_file_array.shape}")
    print(f"🔍 DIMENSION CHECK: data_np.shape = {data_np.shape}")
    
    if pattern_file_array.shape != data_np.shape:
        print(f"🚨 DIMENSION MISMATCH DETECTED!")
        print(f"   Pattern shape: {pattern_file_array.shape}")
        print(f"   Data shape: {data_np.shape}")
        raise ValueError(f"pattern_file_array shape {pattern_file_array.shape} must have the same shape as the data array {data_np.shape}.")
    else:
        print(f"✅ DIMENSION CHECK PASSED: shapes match {pattern_file_array.shape}")

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

def extract_io_number_from_table_name(table_name):
    """
    Extract the IO number from a table name.
    For example: "agate_tt165_m1_IO0_1" -> 0, "agate_tt165_m1_IO77_1" -> 77
    """
    import re
    match = re.search(r'IO(\d+)', table_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None

def get_pattern_for_io_table(table_name, pattern_files):
    """
    Get the appropriate pattern file for an IO table.
    Returns the pattern array for the corresponding IO number.
    Handles tables with fewer than 4 states by adjusting the pattern accordingly.
    """
    io_number = extract_io_number_from_table_name(table_name)
    if io_number is None:
        raise Exception(f"Could not extract IO number from table name: {table_name}")
    
    if io_number < 0 or io_number > 77:
        raise Exception(f"IO number {io_number} is out of range (0-77) for table: {table_name}")
    
    # Get the corresponding pattern file
    pattern_name = f"ecc_2048x32_IO{io_number}"
    file_path = pattern_files.get(pattern_name)
    
    if not file_path:
        raise Exception(f"Pattern file not found for {pattern_name}")
    
    try:
        pattern_array = np.load(file_path)
        print(f"DEBUG: Loaded pattern for {table_name} -> {pattern_name} with shape {pattern_array.shape}")
        
        # The pattern files are already transposed to (2048, 32), so no need to transpose again
        
        # Check for special cases with fewer states (like IO68 with 2 states)
        unique_states = np.unique(pattern_array)
        num_states = len(unique_states)
        
        print(f"DEBUG: Pattern {pattern_name} has {num_states} unique states: {unique_states}")
        
        # Handle tables with fewer than 4 states
        if num_states < 4:
            print(f"DEBUG: Table {table_name} (IO{io_number}) has only {num_states} states - this will be handled properly in the analysis")
            # Note: The analysis functions will automatically handle patterns with fewer states
            # by only processing the states that exist in the pattern
        
        return pattern_array
    except Exception as e:
        raise Exception(f"Error loading pattern file '{file_path}' for table '{table_name}': {str(e)}")


def create_exclude_mask(data_shape, exclude_type, exclude_indices):
    """
    Create a boolean mask for excluding specified rows or columns during analysis.
    
    Args:
        data_shape (tuple): Shape of the data matrix (rows, cols)
        exclude_type (str): Either 'rows' or 'columns'
        exclude_indices (list): List of indices to exclude
    
    Returns:
        numpy.ndarray: Boolean mask where True means include, False means exclude
    """
    try:
        if not exclude_indices:
            return np.ones(data_shape, dtype=bool)  # Include all data
        
        rows, cols = data_shape
        mask = np.ones(data_shape, dtype=bool)
        
        if exclude_type == 'rows':
            # Exclude specified rows
            valid_row_indices = [i for i in exclude_indices if 0 <= i < rows]
            if valid_row_indices:
                mask[valid_row_indices, :] = False
                print(f"Created exclusion mask for {len(valid_row_indices)} rows")
        elif exclude_type == 'columns':
            # Exclude specified columns
            valid_col_indices = [i for i in exclude_indices if 0 <= i < cols]
            if valid_col_indices:
                mask[:, valid_col_indices] = False
                print(f"Created exclusion mask for {len(valid_col_indices)} columns")
        else:
            print(f"Warning: Unknown exclude_type '{exclude_type}', including all data")
            
        return mask
            
    except Exception as e:
        print(f"Error creating exclude mask: {str(e)}")
        return np.ones(data_shape, dtype=bool)  # Include all data on error


def calculate_sigma_distances(data, target_values, table_names, selected_groups=None):
    print("Entering calculate_sigma_distances")
    print("data length:", len(data))
    print("target_values:", target_values)
    print("table_names:", table_names)
    print("selected_groups:", selected_groups)
    
    sigma_distances = {}
    sigma_points = [-4, -3, -2, -1, 0, 1, 2, 3, 4]  # Sigma points to analyze
    
    # If selected_groups is not provided, assume sequential mapping
    if selected_groups is None:
        selected_groups = list(range(len(target_values)))
    
    for table_idx, (table_name, table_data) in enumerate(zip(table_names, data)):
        print(f"Processing table {table_name}")
        sigma_distances[table_name] = []
        
        for state_idx, state_data in enumerate(table_data):
            print(f"Processing state index {state_idx}")
            # Map the state_idx to the actual state number using selected_groups
            if state_idx < len(selected_groups):
                actual_state_number = selected_groups[state_idx]
                print(f"  Actual state number: {actual_state_number}")
                
                # Only process if we have a target value for this actual state number
                if actual_state_number < len(target_values):
                    target = target_values[actual_state_number]
                    mean = np.mean(state_data)
                    std = np.std(state_data)
                    
                    print(f"State {actual_state_number} stats:")
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
                else:
                    print(f"  No target value for state {actual_state_number}, skipping")
            else:
                print(f"  State index {state_idx} exceeds selected_groups length, skipping")
    
    print("Final sigma_distances:", sigma_distances)
    return sigma_distances

def generate_plot(table_names, database_name, form_data):
    import re  # Import re for regex pattern matching
    import os  # Import os for path operations
    print("🚨 FULL FORM_DATA DEBUG:")
    for key, value in form_data.items():
        print(f"  {key}: {value}")
    print("🚨 END FORM_DATA DEBUG")
    color_map_flag = form_data['color_map_flag']  # This is now a boolean
    yanCullinan_flag = form_data.get('yanCullinan_flag', False)  # PLACEHOLDER CHIN EDIT HERE

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
    
    # Get analysis mode for combine analysis
    analysis_mode = form_data.get('analysis_mode', 'individual')
    print("analysis_mode:", analysis_mode)
    
    # Initialize sigma_distances and num_states at the start
    sigma_distances = {}
    num_states = 0
    color_group_keywords = form_data.get('color_group_keywords', [])
    
    print("color_map_flag:", color_map_flag)

    print("target_values:", target_values)  # Print target values for debugging
    print("custom_division:", custom_division)  # Print custom_division for debugging
    print("using_conductance:", using_conductance)  # Print using_conductance for debugging
    print("using_linear_conversion:", using_linear_conversion)  # Print using_linear_conversion for debugging

    print("table_names:", table_names)
    table_names = reorder_tables_fuxi(table_names)
    print("reordered_table_names:", table_names)

    selected_groups = form_data.get('selected_groups', "")
    print("selected_groups:", selected_groups)
    
    # Initialize pattern_file_array to avoid 'referenced before assignment' error
    pattern_file_array = None
    print(f"DEBUG: Initialized pattern_file_array = {pattern_file_array}")
    
    if form_data['state_pattern_type'] == 'predefined':
        # Define the path to your state pattern files directory
        state_pattern = form_data.get('state_pattern')
        print("🔍 DEBUG: state_pattern from form_data:", state_pattern)
        print("🔍 DEBUG: form_data keys:", list(form_data.keys()))
        print("🔍 DEBUG: ecc_pattern_mode:", form_data.get('ecc_pattern_mode', 'not_set'))
        # Define a dictionary to map state patterns to their file paths
        '''pattern_files = {
            "1296x64_rowbar_4states": "State_pattern_files/1296x64_rowbar_4states.npy",
            "2048x32_rowbar_4states": "State_pattern_files/2048x32_rowbar_4states.npy",
            "3x4_4states_debug": "State_pattern_files/3x4_4states_debug.npy",
            "248x248_checkerboard_4states": "State_pattern_files/248x248_checkerboard_4states.npy",
            "1296x64_Adrien_random_4states": "State_pattern_files/1296x64_Adrien_random_4states.npy",
            "248x248_1state": "State_pattern_files/248x248_1state.npy",
            "1296x64_1state": "State_pattern_files/1296x64_1state.npy",
            "248x248_16states": "State_pattern_files/248x248_16states.npy",
            "248x248_2states": "State_pattern_files/248x248_2states.npy",
            "62x62_2states": "State_pattern_files/62x62_2states.npy",
            "248x1_1state": "State_pattern_files/248x1_1state.npy",
            "248x256_1state": "State_pattern_files/248x256_1state.npy",
            "82944x78_ecc_fuxi": "State_pattern_files/82944x78_ecc_fuxi.npy",
            "248x256_1state": "State_pattern_files/248x256_1state.npy",
            "256x32_pr0": "State_pattern_files/256x32_pr0.npy",
            "256x32_pr1": "State_pattern_files/256x32_pr1.npy",
            "test_chin": "State_pattern_files/ecc_new.npy",
            
        }'''

        pattern_files = get_pattern_files()

        # Fetch the file path based on the state pattern using a dictionary lookup
        file_path = pattern_files.get(state_pattern)
        print(f"🔍 DEBUG: Looking up pattern '{state_pattern}' in pattern_files")
        print(f"🔍 DEBUG: Found file_path: {file_path}")

        # Special handling for ECC patterns
        if state_pattern == "ecc_2048x32_78tables":
            print("DEBUG: Processing 78-table pattern - will match each table with its corresponding IO pattern")
            # This will be handled later in the processing loop
            pattern_file_array = "SPECIAL_78TABLES"  # Special marker
        elif state_pattern == "combined_ecc_subset":
            print(f"🔍 DEBUG: Processing ECC subset pattern: {state_pattern}")
            print(f"🔍 DEBUG: ECC pattern mode from form_data: {form_data.get('ecc_pattern_mode', 'not_set')}")
            
            # This is the standard ECC subset case
            if 'ecc_io_numbers' in form_data and form_data['ecc_io_numbers']:
                print(f"🔍 DEBUG: Valid ECC IO numbers found: {form_data['ecc_io_numbers']}")
                pattern_file_array = "SPECIAL_ECC_SUBSET"  # Special marker
            else:
                # No valid IO numbers - this should be an error
                raise Exception(f"ECC subset pattern 'combined_ecc_subset' requested but no valid IO numbers found in session. Please try the ECC pattern detection again.")
        elif state_pattern.startswith("ecc_2048x32_combined_"):
            print(f"🔍 DEBUG: Processing ECC subset pattern: {state_pattern}")
            print(f"🔍 DEBUG: ECC pattern mode from form_data: {form_data.get('ecc_pattern_mode', 'not_set')}")
            
            # Check if user has manually overridden the pattern selection
            # If ecc_pattern_mode is 'normal', it means user switched to manual selection
            if form_data.get('ecc_pattern_mode') == 'normal':
                print(f"🔍 DEBUG: User has switched to manual pattern selection - ignoring ECC combined pattern")
                print(f"🔍 DEBUG: Will treat '{state_pattern}' as regular pattern (should fail if not found)")
                # Don't use ECC special handling - let it fall through to regular pattern loading
                pattern_file_array = None
            elif 'ecc_io_numbers' in form_data and form_data['ecc_io_numbers']:
                print(f"🔍 DEBUG: Valid ECC IO numbers found: {form_data['ecc_io_numbers']}")
                pattern_file_array = "SPECIAL_ECC_SUBSET"  # Special marker
            else:
                # No valid IO numbers - this means user selected non-ECC tables but chose "ECC pattern"
                # This should fail with a clear error rather than proceeding with wrong patterns
                available_io_numbers = []
                for table_name in table_names:
                    match = re.search(r'IO(\d+)', table_name)
                    if match:
                        available_io_numbers.append(int(match.group(1)))
                
                if not available_io_numbers:
                    raise Exception(f"ECC pattern processing requested for non-ECC tables. The selected tables {table_names} do not contain IO patterns (IO0, IO1, etc.) required for ECC analysis. Please select tables with IO patterns or choose 'No, Regular Analysis' in the ECC pattern detection.")
                else:
                    print(f"🔍 DEBUG: Found IO numbers in table names but not in session: {available_io_numbers}")
                    pattern_file_array = "SPECIAL_ECC_SUBSET"  # Special marker
        elif any(re.search(r'IO(\d+)', name) for name in table_names) and len(table_names) > 1:
            # Check if user explicitly chose regular analysis (ecc_pattern_mode = 'normal')
            if form_data.get('ecc_pattern_mode') == 'normal':
                print(f"🚨 DEBUG: SKIPPING auto-detection because ecc_pattern_mode is 'normal' - user chose regular analysis")
                print(f"🚨 DEBUG: Will use manually selected pattern: {state_pattern}")
                print(f"🚨 DEBUG: This should cause dimension mismatch for 2048x32 tables vs 1296x64 pattern")
                # Don't auto-detect ECC - let it fall through to regular pattern loading
                # Set pattern_file_array to None so we'll load the regular pattern file below
                pattern_file_array = None
            else:
                # Auto-detect case: Multiple IO tables (works for both combine and individual analysis)
                print(f"DEBUG: Auto-detecting ECC subset case - Multiple IO tables with pattern '{state_pattern}' selected")
                print(f"DEBUG: Table names: {table_names}")
                print(f"DEBUG: Analysis mode: {analysis_mode}")
                # Check if this is an ECC subset case by looking at table names
                io_numbers = []
                for table_name in table_names:
                    match = re.search(r'IO(\d+)', table_name)
                    if match:
                        io_numbers.append(int(match.group(1)))
                
                if len(io_numbers) > 1:  # Multiple IO tables
                    print(f"DEBUG: Auto-detected ECC subset with IO numbers: {io_numbers}")
                    pattern_file_array = "SPECIAL_ECC_SUBSET"  # Force ECC subset handling for both modes
                else:
                    print(f"DEBUG: Single IO table or no IO pattern detected, using individual pattern")
                    pattern_file_array = "SPECIAL_ECC_SUBSET" if io_numbers else None
        # Load the pattern file array if the file path is found and pattern_file_array is still None
        if file_path and pattern_file_array is None:
            try:
                pattern_file_array = np.load(file_path)
                print(f"🔍 DEBUG: Loaded pattern_file_array with shape {pattern_file_array.shape}")
                print(f"🔍 DEBUG: Pattern file loaded from: {file_path}")
                print(f"🔍 DEBUG: Successfully loaded regular pattern '{state_pattern}'")
                
                # Check for dimension mismatch early when user explicitly chose regular analysis
                if form_data.get('ecc_pattern_mode') == 'normal' and analysis_mode == 'combine':
                    # For combine mode, pattern must match the COMBINED dimensions, not individual table dimensions
                    num_tables = len(table_names)
                    if num_tables > 1:
                        # Get actual table dimensions instead of hardcoding 2048x32
                        print(f"🔍 GETTING ACTUAL TABLE DIMENSIONS...")
                        first_table_name = table_names[0]
                        
                        # Use get_table_dimensions from db_operations to get actual dimensions
                        from db_operations import get_table_dimensions
                        try:
                            first_table_rows, first_table_cols = get_table_dimensions(database_name, first_table_name)
                            print(f"   First table ({first_table_name}): {first_table_rows}x{first_table_cols}")
                            
                            # Calculate expected combined shape based on actual table dimensions
                            expected_combined_shape = (first_table_rows, first_table_cols * num_tables)
                            
                        except Exception as e:
                            print(f"   ⚠️  Could not get table dimensions: {e}")
                            print(f"   ⚠️  Falling back to hardcoded assumption: 2048x32")
                            # Fallback to previous hardcoded logic
                            expected_combined_shape = (2048, 32 * num_tables)
                        
                        print(f"🔍 COMBINE MODE DIMENSION CHECK:")
                        print(f"   Number of tables: {num_tables}")
                        print(f"   Pattern shape: {pattern_file_array.shape}")
                        print(f"   Expected combined data shape: {expected_combined_shape}")
                        
                        # Check if pattern dimensions match combined data dimensions
                        if pattern_file_array.shape != expected_combined_shape:
                            # Check if we can auto-replicate a single-table pattern for multi-table combine
                            single_table_rows, single_table_cols = expected_combined_shape[0], expected_combined_shape[1] // num_tables
                            single_table_shape = (single_table_rows, single_table_cols)
                            
                            if pattern_file_array.shape == single_table_shape:
                                print(f"🔄 AUTO-REPLICATING PATTERN FOR COMBINE MODE:")
                                print(f"   Original pattern shape: {pattern_file_array.shape}")
                                print(f"   Single table dimensions: {single_table_shape}")
                                print(f"   Replicating pattern {num_tables} times to match combined data shape: {expected_combined_shape}")
                                
                                # Replicate the pattern horizontally to match combined data
                                replicated_pattern = np.tile(pattern_file_array, (1, num_tables))
                                pattern_file_array = replicated_pattern
                                print(f"🔍 DEBUG: After replication - pattern_file_array.shape = {pattern_file_array.shape}")
                                
                                print(f"✅ PATTERN REPLICATED: New pattern shape: {pattern_file_array.shape}")
                                print(f"✅ DIMENSION CHECK PASSED: Replicated pattern matches combined data shape")
                            else:
                                print(f"🚨 DIMENSION MISMATCH DETECTED!")
                                print(f"   Pattern shape: {pattern_file_array.shape}")
                                print(f"   Expected single table shape: {single_table_shape}")
                                print(f"   Expected combined shape: {expected_combined_shape}")
                                raise ValueError(f"Pattern dimension mismatch: Selected pattern '{state_pattern}' has shape {pattern_file_array.shape}, but combined data from {num_tables} tables will have shape {expected_combined_shape}. Please select a compatible pattern or use individual analysis mode.")
                        else:
                            print(f"✅ DIMENSION CHECK PASSED: Pattern matches combined data shape")
                # Special handling for 82944x78_ecc_fuxi.npy which is actually (78, 1296, 64)
                if state_pattern == "82944x78_ecc_fuxi":
                    # Reshape the 3D array to 2D (78, 82944) and then transpose to (82944, 78)
                    pattern_file_array = pattern_file_array.reshape(78, 82944).T
                    print(f"DEBUG: Reshaped 82944x78_ecc_fuxi to {pattern_file_array.shape}")
                # Special handling for 65536x78_ecc.npy which is actually (78, 32, 2048)
                elif state_pattern == "65536x78_ecc":
                    # Reshape the 3D array to 2D (78, 65536) and then transpose to (65536, 78)
                    pattern_file_array = pattern_file_array.reshape(78, 65536).T
                    print(f"DEBUG: Reshaped 65536x78_ecc to {pattern_file_array.shape}")
                # Special handling for ecc_2048x32_IO files which are actually (32, 2048)
                elif state_pattern.startswith("ecc_2048x32_IO"):
                    # Transpose from (32, 2048) to (2048, 32) to match expected dimensions
                    if len(pattern_file_array.shape) == 2 and pattern_file_array.shape == (32, 2048):
                        pattern_file_array = pattern_file_array.T
                        print(f"DEBUG: Transposed {state_pattern} from (32, 2048) to {pattern_file_array.shape}")
                    else:
                        print(f"DEBUG: Warning - {state_pattern} has unexpected shape: {pattern_file_array.shape}")
                print(f"DEBUG: Final pattern_file_array after loading: {pattern_file_array is not None}")
            except Exception as e:
                raise Exception(f"Error loading pattern file '{file_path}': {str(e)}")
        elif pattern_file_array is None:
            # Only raise error if no file_path was found AND no special handling was applied
            print(f"🔍 DEBUG: No file_path found for pattern '{state_pattern}'")
            print(f"🔍 DEBUG: Available patterns: {list(pattern_files.keys())}")
            if state_pattern.startswith("ecc_2048x32_combined_"):
                print(f"🔍 DEBUG: This appears to be a combined ECC pattern that user tried to override")
                print(f"🔍 DEBUG: User likely selected manual pattern but form still submitted ECC pattern name")
            raise Exception(f"Pattern file not found for state pattern: {state_pattern}. Available patterns: {list(pattern_files.keys())}")
        else:
            # Pattern was handled by special logic (e.g., SPECIAL_ECC_SUBSET), continue processing
            print(f"🔍 DEBUG: Pattern '{state_pattern}' handled by special logic: pattern_file_array = {pattern_file_array}")
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
    # Handle the case where target_ranges might be None or empty
    if target_ranges:
        target_ranges = [float(x) for x in target_ranges.split(',') if x.replace('.', '', 1).isdigit()]
    else:
        target_ranges = []
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
    filtered_ber_results = []  # Initialize for BER results from CDF analysis

    # Compute the global min and max values among all data matrices
    data_matrices = []
    bitmap_mask_to_save = None  # Will store the bitmap mask if we need to generate one
    
    # Handle combine analysis mode
    if analysis_mode == 'combine':
        print("COMBINE ANALYSIS MODE: Processing all tables as one combined dataset")
        
        # Collect all data matrices first
        combined_data_matrices = []
        original_table_names = table_names.copy()  # Keep original table names for reference
        
        for table_name in table_names:
            data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
            
            # Apply conversions and filters to each table
            # Apply conductance conversion if enabled
            if using_conductance and conductance_params:
                print(f"Converting table {table_name} to conductance values")
                data_matrix = convert_table_to_conductance(data_matrix, conductance_params)
            # Apply linear conversion if enabled
            elif using_linear_conversion:
                # Get conversion parameters from form_data
                conversion_params = {
                    'input_min': form_data.get('linear_input_min', 0),
                    'input_max': form_data.get('linear_input_max', 63),
                    'output_min': form_data.get('linear_output_min', 60),
                    'output_max': form_data.get('linear_output_max', 170)
                }
                print(f"Converting table {table_name} using linear conversion")
                data_matrix = convert_table_to_linear(data_matrix, conversion_params)
            
            # Apply bitmap mask if specified
            apply_bitmap_mask = form_data.get('apply_bitmap_mask', '').strip()
            if apply_bitmap_mask:
                print(f"Applying bitmap mask '{apply_bitmap_mask}' to table {table_name}")
                from route_handlers import load_bitmap_mask, validate_mask_dimensions
                
                mask_array, mask_dimensions = load_bitmap_mask(database_name, apply_bitmap_mask)
                if mask_array is not None:
                    is_valid, validation_message = validate_mask_dimensions(mask_array, data_matrix)
                    if is_valid:
                        data_matrix[mask_array == 0] = np.nan
                        print(f"Bitmap mask applied to {table_name}")
                    else:
                        print(f"Bitmap mask validation failed for {table_name}: {validation_message}")
                else:
                    print(f"Could not load bitmap mask '{apply_bitmap_mask}' for table {table_name}")
            
            # Apply data range and negative value filtering
            data_min_value = form_data.get('data_min_value')
            data_max_value = form_data.get('data_max_value')
            if data_min_value is not None or data_max_value is not None:
                mask = np.ones(data_matrix.shape, dtype=bool)
                if data_min_value is not None:
                    mask &= (data_matrix >= data_min_value)
                if data_max_value is not None:
                    mask &= (data_matrix <= data_max_value)
                data_matrix[~mask] = np.nan
                print(f"Data range filter applied to {table_name}")
            
            filter_negative_values = form_data.get('filter_negative_values', False)
            if filter_negative_values:
                data_matrix[data_matrix < 0] = np.nan
                print(f"Negative value filter applied to {table_name}")
            
            combined_data_matrices.append(data_matrix)
        
        # For combine analysis, create a single "virtual" table that contains all data
        if len(combined_data_matrices) > 1:
            # Concatenate all matrices horizontally (side by side)
            combined_matrix = np.hstack(combined_data_matrices)
            print(f"Combined {len(combined_data_matrices)} tables into matrix shape: {combined_matrix.shape}")
            
            # Create a single table name representing the combination
            combined_table_name = f"Combined_{len(original_table_names)}_tables"
            table_names = [combined_table_name]  # Use single combined table
            data_matrices = [(combined_table_name, combined_matrix)]  # Store as tuple for consistency
            
            # Handle pattern combination for the combined data matrix
            if form_data.get('state_pattern') == "ecc_2048x32_78tables":
                print("COMBINE ANALYSIS: Special handling for ecc_2048x32_78tables pattern")
                # For 78 tables pattern, we need to combine the 78 patterns horizontally too
                if pattern_file_array == "SPECIAL_78TABLES":
                    # Load all 78 individual patterns and combine them
                    combined_patterns = []
                    for i, orig_table_name in enumerate(original_table_names):
                        # Extract IO number from table name
                        import re
                        match = re.search(r'IO(\d+)', orig_table_name)
                        if match:
                            io_num = int(match.group(1))
                            if 0 <= io_num <= 77:
                                io_pattern_name = f"ecc_2048x32_IO{io_num}"
                                pattern_files = get_pattern_files()
                                io_pattern_path = pattern_files.get(io_pattern_name)
                                if io_pattern_path:
                                    try:
                                        io_pattern = np.load(io_pattern_path)
                                        if io_pattern.shape == (32, 2048):
                                            io_pattern = io_pattern.T  # Transpose to (2048, 32)
                                        combined_patterns.append(io_pattern)
                                        print(f"Added 78-table pattern for {io_pattern_name} (shape: {io_pattern.shape})")
                                    except Exception as e:
                                        print(f"Error loading pattern {io_pattern_name}: {e}")
                    
                    if combined_patterns:
                        # Horizontally stack all 78 patterns
                        pattern_file_array = np.hstack(combined_patterns)
                        print(f"Combined 78-table pattern shape: {pattern_file_array.shape}")
                    else:
                        print("Warning: No patterns could be loaded for 78-table ECC")
                        pattern_file_array = None
                        
            elif pattern_file_array == "SPECIAL_ECC_SUBSET" or form_data.get('state_pattern', '').startswith("ecc_2048x32_combined_"):
                print("COMBINE ANALYSIS: Special handling for ECC subset pattern")
                print(f"DEBUG: Original table names order: {original_table_names}")
                # For ECC subset, combine the individual IO patterns
                combined_patterns = []
                for i, orig_table_name in enumerate(original_table_names):
                    # Extract IO number from table name
                    import re
                    match = re.search(r'IO(\d+)', orig_table_name)
                    if match:
                        io_num = int(match.group(1))
                        print(f"DEBUG: Processing table {orig_table_name} -> IO{io_num} at position {i}")
                        if 0 <= io_num <= 77:
                            io_pattern_name = f"ecc_2048x32_IO{io_num}"
                            pattern_files = get_pattern_files()
                            io_pattern_path = pattern_files.get(io_pattern_name)
                            if io_pattern_path and os.path.exists(io_pattern_path):
                                io_pattern = np.load(io_pattern_path)
                                combined_patterns.append(io_pattern)
                                print(f"DEBUG: Added pattern for {io_pattern_name} at position {len(combined_patterns)-1} (shape: {io_pattern.shape})")
                            else:
                                print(f"Warning: Pattern file not found for {io_pattern_name}")
                
                if combined_patterns:
                    # Horizontally stack all IO patterns
                    pattern_file_array = np.hstack(combined_patterns)
                    print(f"Combined ECC subset pattern shape: {pattern_file_array.shape}")
                else:
                    print("Warning: No ECC patterns could be loaded for subset")
                    # Fallback to standard pattern handling
                    pattern_file_array = None
            elif pattern_file_array is not None and not isinstance(pattern_file_array, str):
                # For regular patterns, tile them horizontally to match the combined data matrix
                # BUT ONLY if they haven't been replicated already
                num_tables = len(original_table_names)
                expected_combined_cols = 32 * num_tables  # For 2048x32 tables: 4 tables = 128 cols
                
                if num_tables > 1 and pattern_file_array.shape[1] == 32:
                    # Pattern hasn't been replicated yet - tile it
                    print("COMBINE ANALYSIS: Tiling regular pattern to match combined data matrix")
                    combined_patterns = [pattern_file_array] * num_tables
                    pattern_file_array = np.hstack(combined_patterns)
                    print(f"Tiled pattern {num_tables} times, new shape: {pattern_file_array.shape}")
                elif num_tables > 1 and pattern_file_array.shape[1] == expected_combined_cols:
                    # Pattern already replicated - skip tiling
                    print(f"COMBINE ANALYSIS: Pattern already replicated to correct size: {pattern_file_array.shape}")
                else:
                    print(f"COMBINE ANALYSIS: Using pattern as-is: {pattern_file_array.shape}")
                # If only one table, pattern_file_array stays the same
        else:
            # Single table - convert to tuple format for consistency
            data_matrices = [(table_names[0], combined_data_matrices[0])]
        
        print(f"COMBINE ANALYSIS: Final table_names = {table_names}")
        print(f"COMBINE ANALYSIS: Data matrices shapes = {[dm[1].shape for dm in data_matrices]}")
    
    # Individual analysis mode (original behavior)
    # Skip this loop if we're in combine mode since data is already processed above
    if analysis_mode != 'combine':
        for table_name in table_names:
            data_matrix, data_matrix_size = get_full_table_data(table_name, database_name)
            
            # Apply conductance conversion if enabled
            if using_conductance and conductance_params:
                print(f"Converting table {table_name} to conductance values")
                data_matrix = convert_table_to_conductance(data_matrix, conductance_params)
            # Apply linear conversion if enabled
            elif using_linear_conversion:
                # Get conversion parameters from form_data
                conversion_params = {
                    'input_min': form_data.get('linear_input_min', 0),
                    'input_max': form_data.get('linear_input_max', 63),
                    'output_min': form_data.get('linear_output_min', 60),
                    'output_max': form_data.get('linear_output_max', 170)
                }
                print(f"Converting table {table_name} using linear conversion ({conversion_params['input_min']}-{conversion_params['input_max']} → {conversion_params['output_min']}-{conversion_params['output_max']})")
                data_matrix = convert_table_to_linear(data_matrix, conversion_params)
            
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
        # Ensure pattern_file_array is loaded before using it for column analysis
        print(f"DEBUG: Column analysis - pattern_file_array is None: {pattern_file_array is None}")
        if pattern_file_array is None:
            raise Exception(f"Pattern file array not loaded for column-by-column analysis: {form_data.get('state_pattern')}")
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
        
        # Get table-specific division settings for 1D patterns
        if form_data['state_pattern_type'] == '1D' and 'table_division_settings' in form_data:
            table_settings = form_data['table_division_settings'].get(table_name, {})
            table_custom_division = table_settings.get('custom_division', False)
            table_custom_division_values = table_settings.get('custom_division_values', [])
            print(f"Using table-specific division for {table_name}: custom_division={table_custom_division}, values={table_custom_division_values}")
        else:
            # Fallback to global settings for backward compatibility
            table_custom_division = custom_division
            table_custom_division_values = form_data.get('custom_division_values', [])
            print(f"Using global division for {table_name}: custom_division={table_custom_division}, values={table_custom_division_values}")
        
        if target_range_flag == 0:
            if form_data['state_pattern_type'] == '1D':
                # Modify to use the data matrix directly with table-specific settings
                groups, stats, selected_groups = get_group_data_new_from_matrix(
                    data_matrix, selected_groups, number_of_states, table_custom_division, table_custom_division_values)
            elif form_data['state_pattern_type'] == 'predefined':
                # Special handling for ECC patterns (both 78-table and subset)
                if pattern_file_array == "SPECIAL_78TABLES" or pattern_file_array == "SPECIAL_ECC_SUBSET":
                    # For combine analysis, the pattern was already combined above
                    if analysis_mode == 'combine':
                        # In combine mode, pattern_file_array should already be the combined pattern
                        if pattern_file_array is None or isinstance(pattern_file_array, str):
                            raise Exception("Combined ECC pattern not properly loaded in combine analysis mode")
                        exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                        exclude_ranges = form_data.get('exclude_ranges', [])
                        groups, stats, selected_groups = get_group_data_from_matrix(
                            data_matrix, selected_groups, pattern_file_array, exclude_ranges_type, exclude_ranges)
                    else:
                        # Load the specific pattern for this table (individual analysis)
                        print(f"DEBUG: Individual analysis - loading pattern for table {table_name}")
                        table_pattern_array = get_pattern_for_io_table(table_name, pattern_files)
                        print(f"DEBUG: Individual analysis - pattern shape for {table_name}: {table_pattern_array.shape}")
                        print(f"DEBUG: Individual analysis - pattern unique values for {table_name}: {np.unique(table_pattern_array)}")
                        exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                        exclude_ranges = form_data.get('exclude_ranges', [])
                        groups, stats, selected_groups = get_group_data_from_matrix(
                            data_matrix, selected_groups, table_pattern_array, exclude_ranges_type, exclude_ranges)
                else:
                    # Ensure pattern_file_array is loaded before using it
                    print(f"DEBUG: Before predefined processing (target_range_flag=0) - pattern_file_array is None: {pattern_file_array is None}")
                    if pattern_file_array is None:
                        # Check if this is a regular analysis mode that failed
                        if form_data.get('ecc_pattern_mode') == 'normal':
                            raise ValueError(f"Dimension validation failed: Cannot use pattern '{state_pattern}' with the selected tables. This often occurs when trying to use a pattern with different dimensions than your data. For 2048x32 tables in combine mode, the combined data shape will be (2048, 128). Please select a compatible pattern or use individual analysis mode.")
                        else:
                            raise Exception(f"Pattern file array not loaded for predefined pattern: {state_pattern}")
                    # Modify to use the data matrix directly
                    exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                    exclude_ranges = form_data.get('exclude_ranges', [])
                    groups, stats, selected_groups = get_group_data_from_matrix(
                        data_matrix, selected_groups, pattern_file_array, exclude_ranges_type, exclude_ranges)
        elif target_range_flag == 1:
            if form_data['state_pattern_type'] == '1D':
                # Modify to use the data matrix directly with table-specific settings
                groups, stats, selected_groups, table_miao_ber = get_group_data_latest_from_matrix(
                    target_ranges, data_matrix, selected_groups, number_of_states, table_custom_division, table_custom_division_values)
            elif form_data['state_pattern_type'] == 'predefined':
                # Special handling for ECC patterns (both 78-table and subset)
                if pattern_file_array == "SPECIAL_78TABLES" or pattern_file_array == "SPECIAL_ECC_SUBSET":
                    # For combine analysis, the pattern was already combined above
                    if analysis_mode == 'combine':
                        # In combine mode, pattern_file_array should already be the combined pattern
                        if pattern_file_array is None or isinstance(pattern_file_array, str):
                            raise Exception("Combined ECC pattern not properly loaded in combine analysis mode")
                        exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                        exclude_ranges = form_data.get('exclude_ranges', [])
                        groups, stats, selected_groups, table_miao_ber = get_group_data_1124_2_from_matrix(
                            target_ranges, data_matrix, selected_groups, pattern_file_array, exclude_ranges_type, exclude_ranges)
                    else:
                        # Load the specific pattern for this table (individual analysis)
                        print(f"DEBUG: Individual analysis (target_range) - loading pattern for table {table_name}")
                        table_pattern_array = get_pattern_for_io_table(table_name, pattern_files)
                        print(f"DEBUG: Individual analysis (target_range) - pattern shape for {table_name}: {table_pattern_array.shape}")
                        print(f"DEBUG: Individual analysis (target_range) - pattern unique values for {table_name}: {np.unique(table_pattern_array)}")
                        exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                        exclude_ranges = form_data.get('exclude_ranges', [])
                        groups, stats, selected_groups, table_miao_ber = get_group_data_1124_2_from_matrix(
                            target_ranges, data_matrix, selected_groups, table_pattern_array, exclude_ranges_type, exclude_ranges)
                else:
                    # Ensure pattern_file_array is loaded before using it
                    print(f"DEBUG: Before predefined processing (target_range_flag=1) - pattern_file_array is None: {pattern_file_array is None}")
                    if pattern_file_array is None:
                        # Check if this is a regular analysis mode that failed
                        if form_data.get('ecc_pattern_mode') == 'normal':
                            raise ValueError(f"Dimension validation failed: Cannot use pattern '{state_pattern}' with the selected tables. This often occurs when trying to use a pattern with different dimensions than your data. For 2048x32 tables in combine mode, the combined data shape will be (2048, 128). Please select a compatible pattern or use individual analysis mode.")
                        else:
                            raise Exception(f"Pattern file array not loaded for predefined pattern: {state_pattern}")
                    # Modify to use the data matrix directly
                    exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                    exclude_ranges = form_data.get('exclude_ranges', [])
                    groups, stats, selected_groups, table_miao_ber = get_group_data_1124_2_from_matrix(
                        target_ranges, data_matrix, selected_groups, pattern_file_array, exclude_ranges_type, exclude_ranges)
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
        sigma_distances = calculate_sigma_distances(group_data, target_values, table_names, selected_groups)
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
            return ([], [], None, None, None, None, [], [], [], None, None, {}, 0, [], {}, [], [], [], {}, [])
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
    
    # Initialize filtered_miao_ber
    filtered_miao_ber = {}
    if target_range_flag == 1:
        filtered_miao_ber = {name: miao_ber[name] for name in filtered_table_names if name in miao_ber}
        print(f"Debug in generate_plot: target_range_flag=1, miao_ber keys: {list(miao_ber.keys())}")
        print(f"Debug in generate_plot: filtered_miao_ber: {filtered_miao_ber}")
    else:
        print(f"Debug in generate_plot: target_range_flag=0, no BER data calculated")
    
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


    #Boxplot for Yan TEST
    if yanCullinan_flag:
        for table_name, data_matrix in filtered_data_matrices:
            data_matrix = 649.5 - (2.549*data_matrix)
            encoded_plots.append(plot_2d_yan(data_matrix, title=f"2d Visualization for {table_name}"))
            encoded_plots.append(plot_boxplot_yan(data_matrix, title=f"Colormap for {table_name}"))
            encoded_plots.append(plot_cdf_yan(data_matrix, title=f"CDF for {table_name}"))

            
    # Generate plots for filtered tables
    encoded_plots.append(plot_boxplot(filtered_group_data, filtered_table_names))
    
    # Generate comprehensive metrics table combining all data
    from tools_for_plots import plot_comprehensive_metrics_table

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
    
    # Generate location dots map if enabled
    location_dots_flag = form_data.get('location_dots_flag', False)
    location_dots_value = form_data.get('location_dots_value')
    location_dots_map = None  # Initialize to None
    
    if location_dots_flag and location_dots_value is not None:
        print(f"Generating location dots map for target value: {location_dots_value}")
        location_dots_map = generate_location_dots_map(
            filtered_data_matrices, 
            location_dots_value, 
            title_prefix="Location Map"
        )
        if location_dots_map:
            print("Location dots map generated successfully")
        else:
            print("Failed to generate location dots map")
    

    if len(selected_groups) != 1:
        # Generate plots for BER results and get sorted table names
        (sigma_image,
         ppm_image,
         uS_image,
         additional_image,
         sorted_table_names) = plot_ber_tables(filtered_ber_results, target_x_diff, num_interp_points)

        # Generate comprehensive metrics table with BER data
        comprehensive_table = plot_comprehensive_metrics_table(
            filtered_group_data, 
            filtered_avg_values, 
            filtered_std_values, 
            filtered_table_names, 
            selected_groups, 
            filtered_ber_results
        )
        
        # Add the comprehensive table to the plots
        if comprehensive_table:
            encoded_plots.append(comprehensive_table)
        else:
            print("Failed to generate comprehensive table, using fallback")
    else:
        # When there's only one selected group, generate comprehensive table without BER data
        comprehensive_table = plot_comprehensive_metrics_table(
            filtered_group_data, 
            filtered_avg_values, 
            filtered_std_values, 
            filtered_table_names, 
            selected_groups, 
            None  # No BER results for single group
        )
        
        # Add the comprehensive table to the plots
        if comprehensive_table:
            encoded_plots.append(comprehensive_table)
        else:
            print("Failed to generate comprehensive table, using fallback")
        
        # Initialize variables for single group case
        sorted_table_names = filtered_table_names

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
                [],  # Outlier coordinates (removed)
                None,  # Correlation analysis (removed)
                None,  # Cluster map (removed)
                filtered_sigma_distances,
                num_states,
                filtered_table_names,
                sigma_table,  # Add sigma intersections table
                sigma_points,  # Add sigma points
                filtered_avg_values,  # Add real average values
                filtered_std_values,  # Add real standard deviation values
                filtered_ber_results,  # Add real BER results from CDF analysis
                selected_groups,  # Add selected groups for state names
                location_dots_map,  # Add location dots map
                filtered_group_data)  # Add group data for points count

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
            [],  # Outlier coordinates (removed)
            None,  # Correlation analysis (removed)
            None,  # Cluster map (removed)
            filtered_sigma_distances,
            num_states,
            filtered_table_names,
            sigma_table,  # Add sigma intersections table
            sigma_points,  # Add sigma points
            filtered_avg_values,  # Add real average values
            filtered_std_values,  # Add real standard deviation values
            filtered_ber_results,  # Add real BER results from CDF analysis
            selected_groups,  # Add selected groups for state names
            location_dots_map,  # Add location dots map
            filtered_group_data)  # Add group data for points count

# Add helper functions to work with matrices directly instead of fetching from database

def get_group_data_from_matrix(data_matrix, selected_groups, pattern_file_array, exclude_ranges_type='', exclude_ranges=[]):
    """Modified version of get_group_data_1124 that works with a data matrix directly."""
    # Replace zeros with a small value to avoid issues - use np.where for safer comparison
    data_np = np.where(data_matrix == 0, 0.001, data_matrix)

    groups = []
    groups_stats = []  # List to store statistics for each group
    group_idx_to_position = {}

    # Ensure that pattern_file_array has the same shape as data_np
    print(f"🔍 DIMENSION CHECK: pattern_file_array.shape = {pattern_file_array.shape}")
    print(f"🔍 DIMENSION CHECK: data_np.shape = {data_np.shape}")
    
    if pattern_file_array.shape != data_np.shape:
        print(f"🚨 DIMENSION MISMATCH DETECTED!")
        print(f"   Pattern shape: {pattern_file_array.shape}")
        print(f"   Data shape: {data_np.shape}")
        raise ValueError(f"pattern_file_array shape {pattern_file_array.shape} must have the same shape as the data array {data_np.shape}.")
    else:
        print(f"✅ DIMENSION CHECK PASSED: shapes match {pattern_file_array.shape}")

    # Create exclude mask if exclude ranges are specified
    exclude_mask = create_exclude_mask(data_np.shape, exclude_ranges_type, exclude_ranges)
    print(f"🔍 EXCLUDE RANGES DEBUG:")
    print(f"   exclude_ranges_type: '{exclude_ranges_type}'")
    print(f"   exclude_ranges: {exclude_ranges}")
    print(f"   data shape: {data_np.shape}")
    if exclude_ranges_type and exclude_ranges:
        print(f"Applied exclude mask: {exclude_ranges_type} {exclude_ranges}")
        excluded_count = np.sum(~exclude_mask)
        total_count = exclude_mask.size
        print(f"Excluding {excluded_count} out of {total_count} data points ({excluded_count/total_count*100:.1f}%)")
        # Also show which specific rows/columns are being excluded
        if exclude_ranges_type == 'rows':
            excluded_rows = [i for i in range(data_np.shape[0]) if not exclude_mask[i, 0]]
            print(f"   Excluded row indices: {excluded_rows[:10]}{'...' if len(excluded_rows) > 10 else ''}")
        elif exclude_ranges_type == 'columns':
            excluded_cols = [i for i in range(data_np.shape[1]) if not exclude_mask[0, i]]
            print(f"   Excluded column indices: {excluded_cols[:10]}{'...' if len(excluded_cols) > 10 else ''}")
    else:
        print(f"No exclude ranges specified - including all data points")

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

    # Keep track of original selected_groups for consistent structure
    original_selected_groups = selected_groups.copy()
    
    # Filter out selected_groups that don't exist in the dataset for processing
    available_selected_groups = [g for g in selected_groups if g in group_indices]

    for group_idx in original_selected_groups:
        if group_idx in group_indices:
            # Find positions where pattern_file_array equals the current group index
            group_positions = np.where(pattern_file_array == group_idx)
            total_group_positions = len(group_positions[0])
            print(f"Group {group_idx}: Found {total_group_positions} positions in pattern")
            
            # Apply exclude mask to filter out excluded positions
            valid_positions = []
            excluded_positions_count = 0
            for i, (row, col) in enumerate(zip(group_positions[0], group_positions[1])):
                if exclude_mask[row, col]:  # Only include if not excluded
                    valid_positions.append((row, col))
                else:
                    excluded_positions_count += 1
            
            print(f"Group {group_idx}: {excluded_positions_count} positions excluded by mask, {len(valid_positions)} positions remaining")
            values = [data_np[pos] for pos in valid_positions]
            
            # Filter out NaN values (from data range filtering)
            values_before_nan_filter = len(values)
            values = [v for v in values if not np.isnan(v)]
            print(f"Group {group_idx}: {len(values)} valid values after filtering NaN (was {values_before_nan_filter} before NaN filter)")
            
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
        else:
            # Group doesn't exist in this pattern - add empty data to maintain structure
            print(f"Group {group_idx}: Not available in this pattern, adding empty data")
            groups.append([])  # Empty group
            groups_stats.append((0, 0, 0, 0))  # Default stats
            
        # Map group index to its position in the selected_groups list
        group_idx_to_position[group_idx] = len(groups) - 1

    return groups, groups_stats, original_selected_groups

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
    
    # Filter groups and stats to only include selected states
    filtered_groups = []
    filtered_stats = []
    
    for group_idx in selected_groups:
        if 0 <= group_idx < len(groups):
            filtered_groups.append(groups[group_idx])
            filtered_stats.append(groups_stats[group_idx])
        else:
            print(f"Warning: Selected group {group_idx} is out of range (0-{len(groups)-1})")
    
    print(f"Original groups: {len(groups)}, Selected groups: {selected_groups}, Filtered groups: {len(filtered_groups)}")
    
    return filtered_groups, filtered_stats, selected_groups

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

def get_group_data_1124_2_from_matrix(target_ranges, data_matrix, selected_groups, pattern_file_array, exclude_ranges_type='', exclude_ranges=[]):
    """Modified version of get_group_data_1124_2 that works with a data matrix directly."""
    # Get groups data using existing function
    groups, groups_stats, selected_groups = get_group_data_from_matrix(data_matrix, selected_groups, pattern_file_array, exclude_ranges_type, exclude_ranges)
    
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
                exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                exclude_ranges = form_data.get('exclude_ranges', [])
                groups, stats, selected_groups_col = get_group_data_from_matrix(
                    column_data_reshaped, selected_groups, column_pattern, exclude_ranges_type, exclude_ranges)
            elif target_range_flag == 1:
                # Use the matrix-based function with target ranges for this column
                exclude_ranges_type = form_data.get('exclude_ranges_type', '')
                exclude_ranges = form_data.get('exclude_ranges', [])
                groups, stats, selected_groups_col, table_miao_ber = get_group_data_1124_2_from_matrix(
                    target_ranges, column_data_reshaped, selected_groups, column_pattern, exclude_ranges_type, exclude_ranges)
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
        sigma_distances = calculate_sigma_distances(all_column_group_data, target_values, all_column_names, selected_groups)
        num_states = len(target_values)
    
    # Initialize encoded plots list
    encoded_plots = []
    
    # Generate plots for all columns
    encoded_plots.append(plot_boxplot(all_column_group_data, all_column_names))
    
    # Generate comprehensive metrics table for column analysis
    from tools_for_plots import plot_comprehensive_metrics_table
    
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
        
        # Generate comprehensive metrics table for columns with BER data
        comprehensive_column_table = plot_comprehensive_metrics_table(
            all_column_group_data, 
            all_column_avg_values, 
            all_column_std_values, 
            all_column_names, 
            selected_groups, 
            column_ber_results
        )
        
        if comprehensive_column_table:
            encoded_plots.append(comprehensive_column_table)
        else:
            print("Failed to generate comprehensive column table")
        
        # Create best column lists
        best_top_n = sorted_column_names[:10] if len(sorted_column_names) >= 10 else sorted_column_names
        best_top_n_with_io = [f"Col{col_name.split('_Col')[1]}" if '_Col' in col_name else col_name for col_name in best_top_n]
    else:
        # Generate comprehensive metrics table for columns without BER data
        comprehensive_column_table = plot_comprehensive_metrics_table(
            all_column_group_data, 
            all_column_avg_values, 
            all_column_std_values, 
            all_column_names, 
            selected_groups, 
            None  # No BER results for single group
        )
        
        if comprehensive_column_table:
            encoded_plots.append(comprehensive_column_table)
        else:
            print("Failed to generate comprehensive column table")
            
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
            sigma_points,
            None,
            None,
            None,
            None,
            None,  # location_dots_map (None for column analysis)
            None)  # filtered_group_data (None for column analysis)

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

def generate_location_dots_map(data_matrices, target_value, title_prefix="Location Map"):
    """
    Generate a location map showing all coordinates that match the target value.
    
    Args:
        data_matrices: List of tuples (table_name, data_matrix)
        target_value: Integer value to search for
        title_prefix: Prefix for the plot title
    
    Returns:
        Base64 encoded image string
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from io import BytesIO
        import base64
        
        # Set maximum number of matches to process to prevent hanging
        MAX_MATCHES = 10000
        MAX_MATCHES_PER_TABLE = 2000
        
        # Collect all matching coordinates across all tables
        all_matches = []
        table_colors = get_colors(len(data_matrices))
        total_matches_found = 0
        
        print(f"Searching for target value {target_value} across {len(data_matrices)} tables...")
        
        for i, (table_name, data_matrix) in enumerate(data_matrices):
            # Skip NaN values in the search
            valid_mask = ~np.isnan(data_matrix)
            
            # Find coordinates where value equals target_value
            # Handle both exact matches and close matches (within 0.1 tolerance for floating point)
            matches = np.where(valid_mask & (np.abs(data_matrix - target_value) < 0.1))
            
            num_matches = len(matches[0])
            total_matches_found += num_matches
            
            print(f"Table {table_name}: Found {num_matches} matches")
            
            # If too many matches in this table, sample them
            if num_matches > MAX_MATCHES_PER_TABLE:
                print(f"Too many matches in {table_name} ({num_matches}), sampling {MAX_MATCHES_PER_TABLE}")
                # Randomly sample indices
                sample_indices = np.random.choice(num_matches, MAX_MATCHES_PER_TABLE, replace=False)
                sampled_rows = matches[0][sample_indices]
                sampled_cols = matches[1][sample_indices]
            else:
                sampled_rows = matches[0]
                sampled_cols = matches[1]
            
            # Process the matches (sampled or all if not too many)
            for row_idx, col_idx in zip(sampled_rows, sampled_cols):
                actual_value = data_matrix[row_idx, col_idx]
                all_matches.append({
                    'table': table_name,
                    'row': int(row_idx),
                    'col': int(col_idx),
                    'value': float(actual_value),
                    'color': table_colors[i % len(table_colors)]
                })
                
                # Stop if we've collected enough matches overall
                if len(all_matches) >= MAX_MATCHES:
                    print(f"Reached maximum matches limit ({MAX_MATCHES}), stopping collection")
                    break
            
            # Break out of table loop if we've hit the limit
            if len(all_matches) >= MAX_MATCHES:
                break
        
        print(f"Total matches found: {total_matches_found}, Processing: {len(all_matches)}")
        
        if not all_matches:
            # Create a simple "No matches found" plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No locations found matching value {target_value}', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'{title_prefix} - No Matches Found', fontsize=14, fontweight='bold')
            ax.axis('off')
        else:
            # Determine the maximum dimensions across all tables
            max_rows = max(dm[1].shape[0] for dm in data_matrices)
            max_cols = max(dm[1].shape[1] for dm in data_matrices)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Plot each match as a colored dot
            for match in all_matches:
                ax.scatter(match['col'], match['row'], 
                          c=[match['color']], s=50, alpha=0.7, 
                          label=match['table'] if match['table'] not in [m.get_label() for m in ax.get_children() if hasattr(m, 'get_label')] else "")
            
            # Set up the plot
            ax.set_xlim(-0.5, max_cols - 0.5)
            ax.set_ylim(-0.5, max_rows - 0.5)
            ax.invert_yaxis()  # Invert y-axis so (0,0) is at top-left like a matrix
            ax.set_xlabel('Column Index', fontsize=12)
            ax.set_ylabel('Row Index', fontsize=12)
            
            # Create title with sampling information if applicable
            title_parts = [f'{title_prefix} - Locations with Value {target_value}']
            if total_matches_found > len(all_matches):
                title_parts.append(f'(Showing {len(all_matches)} of {total_matches_found} total matches)')
            else:
                title_parts.append(f'({len(all_matches)} matches found)')
            
            ax.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend if there are multiple tables
            if len(data_matrices) > 1:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # Remove duplicate labels
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='upper right', 
                             bbox_to_anchor=(1, 1), fontsize=10)
            
            # Add summary text
            table_counts = {}
            for match in all_matches:
                table_counts[match['table']] = table_counts.get(match['table'], 0) + 1
            
            summary_text = f"Displaying: {len(all_matches)} matches\n"
            if total_matches_found > len(all_matches):
                summary_text += f"Total found: {total_matches_found}\n"
                summary_text += f"(Sampled due to size)\n"
            summary_text += "\nPer table:\n"
            for table, count in sorted(table_counts.items()):
                summary_text += f"{table}: {count}\n"
            
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Save plot to buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        return encoded_image
        
    except Exception as e:
        print(f"Error in generate_location_dots_map: {str(e)}")
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