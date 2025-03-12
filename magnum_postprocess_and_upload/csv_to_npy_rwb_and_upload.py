"""
This scipt is designed to look for new output data from the magnum then crunch it into npy files, rwb files and upload both to the Webapp.
It is meant to run continuously and only look at new files from when the script started.
It's set up to run from this directory on 192.168.68.215: /home/admin2/webapp_2/magnum_postprocess_and_upload
"""
import os, sys
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import argparse
import paramiko
sys.path.insert(1, r"/home/admin2/webapp_2/postprocess")
import core_post_processing_functions as cf
import create_and_upload as cu
import create_and_upload_rwb_param as curp

def post_process_rwb(file, pattern, meta_data, BL_ST = 0, BL_END = 63, 
                     cond_range = (30, 160)):
    # Initialize empty dictionary to store data
    all_data = {}
    raw_data = np.load(file)
    print('Length of raw data: ', len(raw_data))
    for io, io_data in enumerate(raw_data):
        all_data[io] = io_data

    # Transpose if last dim is not row (wl) index
    if all_data[list(all_data.keys())[0]].shape[-1] != 1296:
        for key, sub_data in all_data.items():
            all_data[key] = sub_data.transpose()

    # Define target pattern for pseudorandom_1
    pr_1 = pd.read_csv(os.path.join('/home/admin2/webapp_2/postprocess/pseudorandom_mlm_pattern_1.csv'),header=None)
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
    print('Decoding 2D data into 1D data for each level')
    data_list = {}
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
    rwb['TESTER'] = meta_data['TESTER']
    rwb['SOCKET'] = meta_data['SOCKET']
    rwb['RUN_NAME'] = meta_data['RUN_NAME']
    rwb['CURRENT_COMMIT'] = meta_data['CURRENT_COMMIT']
    rwb['CURRENT_COMMIT_DATE'] = meta_data['CURRENT_COMMIT_DATE']
    rwb['STEPPING'] = meta_data['STEPPING']
    rwb['DIE_ID'] = meta_data['DIE_ID']
    rwb['MACRO'] = meta_data['MACRO']
    rwb['IO'] = rwb.index
    rwb['TEST_START_DATETIME'] = meta_data['TEST_START_DATETIME']
    rwb['ADC_CAL'] = meta_data['ADC_CAL']
    rwb['TEST_SEQUENCE'] = meta_data['TEST_SEQUENCE']
    rwb['TEST_NAME'] = meta_data['TEST_NAME']

    # Reorder columns
    first_columns_order = ['STEPPING','TESTER','SOCKET','BASE_REV','RUN_NAME','CURRENT_COMMIT','CURRENT_COMMIT_DATE','TEST_START_DATETIME','DIE_ID','MACRO','IO','ADC_CAL','TEST_SEQUENCE','TEST_NAME'] # add ,'RUN_NAME' later
    rwb = rwb[first_columns_order + [col for col in rwb.columns if col not in first_columns_order]]

    # Return RWB
    return rwb

def convert_csv_to_npy(post_process_path, odd_macro, file1, file2=None, generate_plot=False, io_list = range(0,78)):
    # Use the filename without the '2' in it
    if 'Finetune2' not in file1:
        file1_name = os.path.basename(file1)
        output_file = os.path.splitext(file1_name)[0]
    else:
        file2_name = os.path.basename(file1)
        output_file = os.path.splitext(file2_name)[0]

    # Replace macro pairs with actual macro for output file
    match = re.search(r'readM(\d+)', file1_name)
    if match:
        result = match.group(1)
        macro = int(result[0])
        # macro_idx0 = int(result[0])
        # macro_idx1 = int(result[1])
    else:
        print("No macro match found from file name.")
    # if odd_macro:
    #     output_file = output_file.replace(f'M{result}',f'M{macro_idx1}')
    # else:
    #     output_file = output_file.replace(f'M{result}',f'M{macro_idx0}')
    output_file = output_file.replace(f'M{result}',f'M{macro}')
    output_path = os.path.join(post_process_path, output_file)

    # Load the CSV file(s)
    data1 = np.genfromtxt(file1, delimiter=',', usecols=range(0, 256), dtype=str)
    if file2 is not None:
        data2 = np.genfromtxt(file2, delimiter=',', usecols=range(0, 256), dtype=str)

    # Convert hex string to integers
    hex_to_int = np.vectorize(lambda x: int(x, 16))

    # Apply the conversion to the loaded data
    data_int1 = hex_to_int(data1)
    if file2 is not None:
        data_int2 = hex_to_int(data2)
        data_int = np.vstack((data_int1, data_int2))
        totWL= 1296
    else:
        data_int = data_int1	# for just first file
        totWL= 648
    array = data_int.reshape(-1, 64, 2, 2, 32)
    array = array.T

    #totWL= 960	# 960 for single 3/4 of macro file for debug

    combined_array = np.zeros((2,2,64,totWL), dtype=object)
    adc_array = np.zeros((2,2,39,64,totWL), dtype=np.uint8)

    print('Extracting data from CSV files')
    for u in range(2):
        for m in range(2):
            for bl in range(64):
                    for wl in range(totWL):
                        for i in range(32):
                            combined_array[u,m,bl,wl] |= (int(array[i,u,m,bl,wl]) << i*8)

                        for io in range(39):
                            adc_array[u,m,io,bl,wl] = np.uint8((combined_array[u,m,bl,wl] >> (6*io)) & 0x3f)

    # io_slice_order = list(range(2,12)) + list(range(14,25)) + list(range(27,38)) + [0,1] + [12,13] + [25,26] + [38]

    # Generate plot if desired
    if generate_plot:
        rows, cols = 8, 10
        fig, axes = plt.subplots(rows, cols, figsize=(24, 20))
        axes_flat = axes.flatten()
        for i in range(39):
            axes_flat[i].imshow(adc_array[0, int(odd_macro),i,:,:], cmap='viridis', interpolation = None, vmin=0, vmax=63, aspect='auto')
            axes_flat[i].set_title(f"IO_Slice_Idx{i}", fontsize=10) 
            # axes_flat[i].axis('off')	# Turn off the axis
        for i in range(39,78):
            axes_flat[i].imshow(adc_array[1, int(odd_macro),i-39,:,:], cmap='viridis', interpolation = None, vmin=0, vmax=63, aspect='auto')
            axes_flat[i].set_title(f"IO_Slice_Idx{i}", fontsize=10) 

        for i in range(78, len(axes_flat)):
            fig.delaxes(axes_flat[i])
        plt.tight_layout()
        print('saving fig to ', output_path + '.png')
        plt.savefig(output_path + '.png')
        plt.close()

    # Save macro-level numpy file
    print('Generating npy file by IO')
    # np.save(output_path + '_adcarray.npy', adc_array) # For debug
    macro_bitlevel = np.empty((0,adc_array.shape[-2], adc_array.shape[-1])) # Initialize empty array

    for io in io_list: # Iterate through each IO and save embedded array of bit map
        # print(macro_bitlevel.shape)
        if io < 39:
            macro_bitlevel = np.concatenate((macro_bitlevel, [adc_array[0, int(odd_macro), io, :, :]]), axis=0)
        else:
            macro_bitlevel = np.concatenate((macro_bitlevel, [adc_array[1, int(odd_macro), io - 39, :, :]]), axis=0)

    print('Saving file')
    numpy_file_path = output_path + '.npy'
    np.save(numpy_file_path, macro_bitlevel)

    return numpy_file_path

def magnum_rawdata_files_list():
    # Server details
    hostname = "192.168.68.91"
    port = 22
    username = "administrator"
    password = "P@$$word"
    directory = "/D:/MagDatalog/Flint/DoE"

        # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        ssh.connect(hostname=hostname, port=port, username=username, password=password)

        # Open an SFTP session
        sftp = ssh.open_sftp()

        # List files in the specified remote directory
        files = sftp.listdir_attr(directory)

        # Prepare the result list
        file_info_list = []
        file_list = []

        for file in files:
            file_name = file.filename
            creation_time = datetime.fromtimestamp(file.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            file_info_list.append((file_name, creation_time))
            file_list.append(file_name)

        # Close the SFTP session
        sftp.close()

        # return file_info_list
        return file_list

    finally:
        # Always close the SSH connection
        ssh.close()


def download_files_from_magnum(file_list, remote_dir, local_dir):
    """
    Download specified files from a remote server via SFTP.
    
    Args:
        host (str): The IP address or hostname of the server.
        port (int): The SSH port (default is 22).
        username (str): The username to authenticate with.
        password (str): The password for authentication.
        remote_dir (str): The directory on the server where files are located.
        file_list (list): List of filenames to download.
        local_dir (str): The local directory where files will be saved.
    """
    # Server details
    hostname = "192.168.68.91"
    port = 22
    username = "administrator"
    password = "P@$$word"

    try:
        # Connect to the server
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=hostname, port=port, username=username, password=password)

        # Initialize SFTP session
        sftp = ssh.open_sftp()

        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        for file_name in file_list:
            remote_path = os.path.join(remote_dir, file_name)
            local_path = os.path.join(local_dir, file_name)

            # Download the file
            sftp.get(remote_path, local_path)
            # print(f"Downloaded: {file_name} -> {local_path}")

        # Close connections
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"An error occurred: {e}")

def push_files_to_magnum(file_list, remote_dir, local_dir):
    """
    Upload specified files to a remote server via SFTP.
    
    Args:
        file_list (list): List of filenames to upload.
        remote_dir (str): The directory on the server where files will be saved.
        local_dir (str): The local directory where files are located.
    """
    # Server details
    hostname = "192.168.68.91"
    port = 22
    username = "administrator"
    password = "P@$$word"

    try:
        # Connect to the server
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=hostname, port=port, username=username, password=password)

        # Initialize SFTP session
        sftp = ssh.open_sftp()

        # Creates remote directory structure if it doesn't exist
        directories = remote_dir.strip("/").split("/")
        current_path = ""
        
        for directory in directories:
            current_path = f"{current_path}/{directory}"
            try:
                sftp.listdir(current_path)
            except IOError:
                sftp.mkdir(current_path)

        # Push each file one by one
        for file_name in file_list:
            local_path = os.path.join(local_dir, file_name)
            remote_path = os.path.join(remote_dir, file_name)

            # Upload the file
            sftp.put(local_path, remote_path)
            print(f"Uploaded: {file_name} -> {remote_path}")

        # Close connections
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"An error occurred: {e}")

def save_list_to_txt(file_path, data_list):
    """
    Save a list of strings to a text file, one item per line.
    
    Args:
        file_path (str): The file path to save the list.
        data_list (list): The list to save.
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")

def load_list_from_txt(file_path):
    """
    Load a list of strings from a text file.
    
    Args:
        file_path (str): The file path to load the list from.
        
    Returns:
        list: The loaded list of strings.
    """
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


# Default flow for this script
if __name__ == '__main__':
    # Global variables to control data processing
    upload_npy_files_to_webapp = True
    if upload_npy_files_to_webapp==False:
        print('Disabling npy file uploads to Webapp')

    upload_rwb_files_to_webapp = True
    if upload_rwb_files_to_webapp==False:
        print('Disabling rwb file uploads to Webapp')
    rwb_db_index = 3

    # Location of folders to process data from and save crunched results
    data_dump_dir = r'/home/admin2/webapp_2/magnum_postprocess_and_upload/Data_dump_temp' # Change this to the proper directory on the magnum tester
    post_process_path = r'/home/admin2/webapp_2/magnum_postprocess_and_upload/Webapp_processed_temp' # Change this to the proper directory on the magnum tester

    # Look for optinal input argument that specifies the date and time after which to look for new files
    parser = argparse.ArgumentParser(description="Optional input to tell which datetime stamp after which to post process files in format YYMMDD_HHMMSS")
    parser.add_argument('-start_datetime', '--start_datetime', type=str, help="Optional input to tell which datetime stamp after which to post process files", default=None)
    args = parser.parse_args()

    user_input = False
    if args.start_datetime:
        print(f"Post processing files from this date as specified by user: {args.start_datetime}")
        script_start_datetime = args.start_datetime
        user_input = True
    else:
        # Record test start time
        script_start_obj = datetime.now()
        script_start_datetime = script_start_obj.strftime('%y%m%d_%H%M%S')

    # # FOR DEBUG ONLY!!! ######
    # script_start_datetime = '250101_000000'
    # #########################

    # List to keep track of processed files in this session
    processed_files = []

    # Load list of all processed files in the output folder, corresponds to first_file path
    # previously_processed_files = [os.path.join(data_dump_dir, x.replace('_rwb.csv', '.csv')) for x in os.listdir(post_process_path) if x.endswith('_rwb.csv')]
    previously_processed_files = load_list_from_txt(r'/home/admin2/webapp_2/magnum_postprocess_and_upload/prev_processed_csv_files.txt')
    for file in previously_processed_files:
        second_file = file.replace('Finetune','Finetune2')
        processed_files.append(file)
        processed_files.append(second_file)

    # Continuously look for new output files
    while True:
        print(f'Looking for files created since {script_start_datetime}')
        time.sleep(5) # Sleep a few seconds each iteration to not consume too much memory or CPU resources

        # Find the latest two output csv files, saved as full path
        # file_list = os.listdir(data_dump_dir)
        # file_list is a tuple of (file name, creation date)
        file_list = magnum_rawdata_files_list()
        # file_list = [os.path.join(data_dump_dir, x) for x in file_list]

        # See which files have been created since the start of the program and their teststart datetime
        files_to_process = []
        for file in file_list:
            # file_datetime = os.path.splitext('_'.join(os.path.basename(file).split('_')[-2:]))[0]
            file_datetime = os.path.splitext('_'.join(file.split('_')[-2:]))[0]
            # ### DBUG
            # if 'readFinetune' in file:
            #     print(f'DEBUG: file={file}, datetime={file_datetime}')
            # print(f'***File \'{os.path.basename(file)}\' extrated datetime is {file_datetime}')
            if (file_datetime > script_start_datetime) and (file not in processed_files):
                # ### DEBUG
                # print(f' - APPENDED: {file}')
                files_to_process.append((file, file_datetime))

        # Sort by creation time: oldest first, keep only macro readouts and print out names
        # files_to_process = sorted(files_to_process, key=lambda d: os.path.getctime(d), reverse=False)
        files_to_process = sorted(files_to_process, key=lambda x: x[1], reverse=False)
        # ### DEBUG
        # print('FILES TO PROCESS1')
        # print(files_to_process)
        files_to_process = [x for x in files_to_process if re.search('readM.*Finetune', x[0])]
        # ### DEBUG
        # print('FILES TO PROCESS2')
        # print(files_to_process)
        if len(files_to_process) > 0:
            print('Found new files to process:')
            for file, file_datetime in files_to_process:
                print(f' - {file}')

        # Loop through each new file to find the complimentary file for that macro readout
        for file, file_datetime in files_to_process:
            ### DEBUG
            # print(f'Looping file: {file}')
            
            # Find matching file and reference first and second files according to number after 'readFinetune'
            # Set the expected names for the paired files
            first_file = None
            second_file = None
            both_files_found = False
            if re.search('Finetune2', file):
                first_file = file.replace('Finetune2','Finetune')
                second_file = file
            else:
                first_file = file
                second_file = file.replace('Finetune','Finetune2')

            # # DEBUG
            # print(f'first_file: {first_file}')
            # print(f'second_file: {second_file}')
            # set_to_print = {file_name for file_name, _ in files_to_process}
            # print(f'set: {set_to_print}')

            # See if both paired readouts exist in the list of files, process the data
            if set([first_file,second_file]) <= {file_name for file_name, _ in files_to_process}:
                print(f'Processing files {first_file} and {second_file}')

                # Add delay to ensure detected CSV file is done writing
                write_verify_delay = 10
                print(f'Delaying processing by {write_verify_delay} seconds to ensure writing is complete')
                time.sleep(write_verify_delay)

                # Reformat test file output datetime to match database format
                # file_datetime = os.path.splitext('_'.join(os.path.basename(first_file).split('_')[-2:]))[0]
                file_dt_obj = datetime.strptime(file_datetime, '%y%m%d_%H%M%S')
                file_metadata_datetime = file_dt_obj.strftime('%Y_%m_%d-%H_%M_%S')

                # Extract socket and test sequence from file name
                extracted_substring = first_file.split('_')[-3]
                socket = extracted_substring[0]
                test_sequence = int(extracted_substring[1:])

                # Extract macro from TestData file output with the matched sequence test
                testdata_file = os.path.splitext(re.sub(r'read.*Finetune', 'TestData', first_file))[0]
                testdata_file = testdata_file.replace(f'__00_{socket}{test_sequence}',f'__00_A{test_sequence}') + '.txt'
                # testdata_file = os.path.join(data_dump_dir, testdata_file) # Define TestData file name for the given read output being analyzed

                # Transfer the files from magnum03 tester to local dir for processing
                download_files_from_magnum([first_file, second_file], remote_dir="/D:/MagDatalog/Flint/DoE", local_dir=data_dump_dir)
                
                # Generate full paths for first, second and testdata files
                first_file_path = os.path.join(data_dump_dir,first_file)
                second_file_path = os.path.join(data_dump_dir,second_file)
                # testdata_file_path = os.path.join(data_dump_dir,testdata_file)
                
                # # Open TestData file to extract macro
                # with open(testdata_file_path, 'r') as file:
                #     matching_lines = [line.strip() for line in file if 'macro=' in line]
                # extracted_macro_list = []
                # for line in matching_lines:
                #     groups = re.search('macro=(\d+)', line)
                #     extracted_macro_list.append(int(groups[1]))
                # if len(set(extracted_macro_list))==1:
                #     macro = extracted_macro_list[0]
                # else:
                #     raise Exception('Multiple macros found in TestData test file for this readout')

                # Extract macro, can be two or one macros depending on name
                group = re.search('readM(\d+)Finetune', first_file)
                if len(group[1])==2:
                    macro_idx0 = int(group[1][0])
                    macro_idx1 = int(group[1][1])
                    macro_list = [macro_idx0, macro_idx1]
                elif len(group[1])==1:
                    macro_idx0 = int(group[1])
                    macro_list = [macro_idx0]
                # ### DEBUG
                # print(f'Macro: {macro}')

                # Do post processing for both macros
                for macro in macro_list:
                    print(f'Processing Macro{macro}')
                    odd_macro = False if (macro%2==0) else True

                    # Start processing by converting csv to npy file
                    print('Converting CSV to npy files')
                    npy_file_path = convert_csv_to_npy(post_process_path, odd_macro, first_file_path, second_file_path, generate_plot=False)
                    # DEBUG
                    # npy_file_path = r"C:\Users\AdrienPierre\Documents\F2MT-285_Magnum_data_processing\Webapp_uploads\Flint_DoE_readFinetune__00_G2_250228_145421.npy"

                    # Set metadata for RWB file
                    metadata = {}
                    metadata['BASE_REV'] = 'na'
                    metadata['TESTER'] = 'MAGNUM03'
                    metadata['RUN_NAME'] = 'na'
                    metadata['CURRENT_COMMIT'] = 'na'
                    metadata['CURRENT_COMMIT_DATE'] = 'na'
                    metadata['STEPPING'] = 'FLINT'
                    metadata['DIE_ID'] = 'na' # Need way to map socket to DIE_ID for a given test
                    metadata['SOCKET'] = socket
                    metadata['MACRO'] = macro # Need to replace with extracted macro from file name
                    metadata['TEST_START_DATETIME'] = file_metadata_datetime
                    metadata['ADC_CAL'] = 'Yes'
                    metadata['TEST_SEQUENCE'] = test_sequence
                    metadata['TEST_NAME'] = 'na'

                    # Create database name: string format must have no spaces and five underscores, keep datetime after last underscore
                    file_datetime_dbformat = file_dt_obj.strftime('%Y%m%d%H%M%S')
                    rawdata_db_name = f"maguser_FLINT_{socket}_{macro}_na_na_{file_datetime_dbformat}"
                    if cu.validate_database_name(rawdata_db_name) is False:
                        raise Exception('Invalid DB name for raw data, please change name')

                    # Post process from npy into rwb dataframe
                    rwb_df = post_process_rwb(npy_file_path, pattern='rowbar', meta_data=metadata)

                    # Save RWB dataframe as a csv file
                    rwb_file_name = os.path.basename(npy_file_path).replace('.npy','_rwb.csv')
                    rwb_df.to_csv(os.path.join(post_process_path, rwb_file_name), index=False)

                    # Generate by-IO NQ plot and save
                    print('Plotting by-IO data')
                    cf.rwb_overlay_levels_nqplot(rwb_df, interactive_mode=False)
                    byIO_plot_name = f"{file_datetime}_M{macro}_{socket}{test_sequence}_NQprob_byIO.png"
                    plt.savefig(os.path.join(post_process_path, byIO_plot_name), dpi=300, bbox_inches="tight")
                    plt.close()

                    print('Plotting overlaid IO data')
                    overlay_IO_plot_name = f"{file_datetime}_M{macro}_{socket}{test_sequence}_NQprob_overlayIO.png"
                    cf.rwb_groupby_level_nqplot(rwb_df, plot_level=-2, legend=False, title=overlay_IO_plot_name.replace('.png',''), interactive_mode=False)
                    plt.savefig(os.path.join(post_process_path, overlay_IO_plot_name), dpi=300, bbox_inches="tight")
                    plt.close()

                    print('Plotting macro average data')
                    rwb_mean = rwb_df.groupby(['SOCKET','TEST_START_DATETIME','DIE_ID','MACRO','TEST_NAME','TEST_SEQUENCE'])[[x for x in rwb_df.columns if 'ADC_PPM' in x]].mean().reset_index()
                    macroavg_plot_name = f"{file_datetime}_M{macro}_{socket}{test_sequence}_NQprob_macroavg.png"
                    cf.rwb_groupby_level_nqplot(rwb_mean, plot_level=-2, legend=False, title=macroavg_plot_name.replace('.png',''), interactive_mode=False, overlay_col='TEST_SEQUENCE')
                    plt.savefig(os.path.join(post_process_path, macroavg_plot_name), dpi=300, bbox_inches="tight")
                    plt.close()

                    # Push post processed png and rwb files to server
                    push_files_to_magnum([byIO_plot_name, overlay_IO_plot_name, macroavg_plot_name, rwb_file_name], remote_dir=f"/D:/MagDatalog/Flint/Postprocessed_results/{file_datetime}", local_dir=post_process_path)

                    # Upload npy files
                    if upload_npy_files_to_webapp:
                        print('Uploading numpy files for each IO')
                        npy_macro_data = np.load(npy_file_path)
                        for io, io_data in enumerate(npy_macro_data):
                            io_data = np.transpose(io_data)
                            io_webapp_file_name = f'{file_datetime_dbformat}_{socket}{test_sequence}_M{macro}_IO{io}'
                            cu.upload_to_db(df=io_data, table_name=io_webapp_file_name, DATABASE_NAME=rawdata_db_name,
                                                    test_start_datetime=file_metadata_datetime)
                            
                    # Upload rwb files
                    if upload_rwb_files_to_webapp:
                        curp.upload_to_db(rwb_df, f'rwb_db_{rwb_db_index}', 'rwb')

                    # Delete npy file to save space
                    os.remove(npy_file_path)

                # Remove processed files from list of files to process
                files_to_process.remove((first_file, file_datetime))
                files_to_process.remove((second_file, file_datetime))

                # Add processed files to list of processed files
                processed_files.append(first_file)
                processed_files.append(second_file)
                previously_processed_files.extend([first_file,second_file])
                save_list_to_txt(r'/home/admin2/webapp_2/magnum_postprocess_and_upload/prev_processed_csv_files.txt', previously_processed_files)

        if user_input and (len(files_to_process)==0):
            print('End of start datetime override mode, please rerun this script without an input argument to resume continuous post processing')
            break
