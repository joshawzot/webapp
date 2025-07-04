# Manually recrunch RWB from raw npy files
import os, sys
import re
import pandas as pd
import numpy as np
import core_post_processing_functions as cf
from datetime import datetime

def post_process_rwb(file, io, pattern, meta_data, postprocess_path, test_name, BL_ST = 0, BL_END = 63,
                     cond_range = (30, 160)):
    # Initialize empty dictionary to store data
    all_data = {}
    if io == -1: # full macro readout
        raw_data = np.load(file)
        for io_num, io_data in enumerate(raw_data):
            all_data[io_num] = io_data
    else: # Single-io readout
        all_data[test_name] = np.load(file)

    # Transpose if last dim is not row (wl) index
    if all_data[list(all_data.keys())[0]].shape[-1] != 1296:
        for key, sub_data in all_data.items():
            all_data[key] = sub_data.transpose()

    # Define target pattern for pseudorandom_1
    pr_1 = pd.read_csv(os.path.join(os.path.dirname(postprocess_path),'pseudorandom_mlm_pattern_1.csv'),header=None)
    pseudorandom_1_162wl = pr_1.to_numpy() # Base random pattern is 162 WLs, need to *8 to get to 1296
    target_pattern_pseudorandom_1 = np.concatenate([pseudorandom_1_162wl]*8, axis=1)

    # Define target pattern for rowbar
    target_pattern_rowbar = np.zeros((64,1296))
    target_pattern_rowbar[:,324:324*2] = 1
    target_pattern_rowbar[:,324*2:324*3] = 2
    target_pattern_rowbar[:,324*3:] = 3

    # Define binary cycle: bottom half of BLs is reset, top half is set
    target_pattern_cyclebinary = np.zeros((64,1296))
    bl_range = BL_END - BL_ST + 1
    target_pattern_cyclebinary[:BL_END - int(bl_range/2) + 1,:] = 3 # Set bottom half of array to level3, e.g. for default BL range: 63 - 32 + 1 = 32

    # Set read pattern
    if pattern == 'pr1':
        pattern_map = target_pattern_pseudorandom_1
    elif pattern == 'rowbar':
        pattern_map = target_pattern_rowbar
    elif pattern == 'pr1bin':
        pattern_map = np.where(target_pattern_pseudorandom_1 >= 2, 0, 3)
    elif pattern == 'invrowbar':
        pattern_map = np.where(target_pattern_rowbar >= 2, 0, 3)
    elif pattern == 'cyclebinary':
        pattern_map = target_pattern_cyclebinary
    elif bool(re.search('^level[0-3]$', pattern)):
        match = re.search('^level(\d+)', pattern)
        level = int(match.group(1))
        pattern_map = np.zeros((64,1296)) + level
        

    # Iterate through each key and decode levels 0-3
    data_list = {}
    print(f'Decoding 2D data to levels')
    for key in all_data.keys():
        data_list[key] = {}
        for level in [0,1,2,3]:
            # Mask the 2d array to the level of interest in this loop iteration
            level_mask = (pattern_map == level)
            rowbar_2d_temp = np.where(level_mask, all_data[key], np.nan)
            # Trim any data from BLs outside the region of interest
            rowbar_2d_temp = rowbar_2d_temp[BL_ST:BL_END + 1, :]
            # Flatten into a linear array and exlude np.nan values
            linear_temp = rowbar_2d_temp.flatten()
            data_list[key][level] = (linear_temp[~np.isnan(linear_temp)])
            # Save as array of -1 if there's nothing from this level
            if len(data_list[key][level])==0:
                data_list[key][level] = np.zeros(1000)-1

    # Run all readouts through rwb cruncher
    print('Crunching RWB')
    rwb = cf.rwb_intcond_overlaylevelnqplot(data_list, list(data_list.keys()), cond_range = cond_range)

    # Print standard run info into all rows
    rwb['BASE_REV'] = meta_data['BASE_REV']
    rwb['RUN_NAME'] = meta_data['RUN_NAME']
    rwb['CURRENT_COMMIT'] = meta_data['CURRENT_COMMIT']
    rwb['CURRENT_COMMIT_DATE'] = meta_data['CURRENT_COMMIT_DATE']
    rwb['STEPPING'] = meta_data['STEPPING']
    rwb['DIE_ID'] = meta_data['DIE_ID']
    rwb['MACRO'] = meta_data['MACRO']
    rwb['TEST_START_DATETIME'] = meta_data['TEST_START_DATETIME']
    rwb['ADC_CAL'] = meta_data['ADC_CAL']

    if io == -1: # Full macro read
        rwb['IO'] = rwb.index
        rwb['TEST_NAME'] = test_name
    else: # Individual IO read
        rwb['IO'] = io
        rwb['TEST_NAME'] = rwb.index # Index is the same as input test_name

    # Reorder columns
    first_columns_order = ['STEPPING','BASE_REV','RUN_NAME','CURRENT_COMMIT','CURRENT_COMMIT_DATE','TEST_START_DATETIME','DIE_ID','MACRO','IO','ADC_CAL','TEST_NAME'] # add ,'RUN_NAME' later
    rwb = rwb[first_columns_order + [col for col in rwb.columns if col not in first_columns_order]]

    # Return RWB
    return rwb

# Find all .npy files
root_dir = r"C:\Users\AdrienPierre\Documents\debug\Full macro readout"
folders = ['Full macro readout'] #os.listdir(root_dir)
raw_data_format = 'macro_read' # can be 'cf_bitlevel_pull', 'macro_read', or 'numpy'

metadata = {}
df_master = pd.DataFrame()
if raw_data_format == 'macro_read':
    # Print standard run info into all rows
    metadata['BASE_REV'] = 'FLINT_POR_REV6'
    metadata['RUN_NAME'] = 'PSTFORM' # folder.split('_')[5]
    metadata['CURRENT_COMMIT'] = 'na'
    metadata['CURRENT_COMMIT_DATE'] = 'na'
    metadata['STEPPING'] = 'FLINT'
    metadata['DIE_ID'] = 'TT30'
    metadata['TEST_START_DATETIME'] = '2025_03_05-10_00_00'
    metadata['ADC_CAL'] = 'No'
    metadata['TEST_NAME'] = 'POSTFORM'

    files = [x for x in os.listdir(os.path.join(root_dir)) if x.endswith('.npy')]
    for file in files:
        metadata['MACRO'] = int(file.split('Macro')[1].replace('.npy',''))

        # Load file and crunch
        file_path = os.path.join(root_dir,file)
        df = post_process_rwb(file_path, -1, 'level0', metadata, '', metadata['TEST_NAME'])
        # df_master = pd.concat([df_master,df], ignore_index=True)

        # Save file
        df.to_csv(f'manual_RWB_crunch_output_{metadata['TEST_NAME']}_macro{metadata['MACRO']}.csv', index=False)

elif raw_data_format == 'numpy':
    for folder in folders:
        print(folder)
        # Print standard run info into all rows
        metadata['BASE_REV'] = 'FLINT_POR_REV5'
        metadata['RUN_NAME'] = 'EXTPPMTST' # folder.split('_')[5]
        metadata['CURRENT_COMMIT'] = '5d430abb9bcc3b94eecd936a4a3e16d48433215e'
        metadata['CURRENT_COMMIT_DATE'] = '2025-02-06 17:31:39'
        metadata['STEPPING'] = 'FLINT'
        metadata['DIE_ID'] = 'TT15' # folder.split('_')[7].upper()
        metadata['MACRO'] = 1 # int(folder.split('_')[8].replace('macro',''))
        metadata['TEST_START_DATETIME'] = '2025_02_06-16_35_04'
        metadata['ADC_CAL'] = 'Yes'

        files = [x for x in os.listdir(os.path.join(root_dir,folder)) if x.endswith('.npy')]
        for file in files:
            file_path = os.path.join(root_dir,folder,file)
            metadata['IO'] = int(file.split('_')[3].replace('io',''))
            metadata['TEST_NAME'] = '_'.join(file.split('_')[4:6]).replace('.npy','')

            # Load file and crunch
            df = post_process_rwb(file_path, metadata['IO'], 'pr1', metadata, '', metadata['TEST_NAME'])
            df_master = pd.concat([df_master,df], ignore_index=True)

    df_master.to_csv('manual_RWB_crunch_output.csv', index=False)

elif raw_data_format == 'cf_bitlevel_pull':
    list_files = os.listdir(root_dir)
    for file in list_files:
        file_path = os.path.join(root_dir, file)
        df = pd.read_csv(file_path)
        for sub_df in df.groupby(['DIE_ID','MACRO','IO']):
            metadata['CURRENT_COMMIT'] = 'NA'
            metadata['CURRENT_COMMIT_DATE'] = 'NA'
            metadata['BASE_REV'] = 'FLINT_POR_REV5'
            metadata['STEPPING'] = 'FLINT'
            metadata['DIE_ID'] = df.DIE_ID
            metadata['MACRO'] = df.MACRO
            metadata['IO'] = df.IO
            metadata['ADC_CAL'] = 'Yes'
            data_cols = [x for x in sub_df.columns if x.startswith('ADC_')]
            for col in data_cols:
                metadata['RUN_NAME'] = col.split('_')[1]
                metadata['TEST_NAME'] = '_'.join(col.split('_')[2,3])
                date_str = col.split('_')[4]
                dt = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                metadata['TEST_START_DATETIME'] = dt.strftime("%Y_%m_%d-%H_%M_%S")
