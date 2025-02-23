import pandas as pd
import re, os
import datetime
from pathlib import Path
import argparse
import json

def save_copy_main_prog_c(save_dir, dir_path = None):
    if dir_path is None:
        dir_path = '../bare_metal_c_interface/slate_c_mlm_finetuning/main_prog.c'

    # Open the text file and read its content
    with open(dir_path, 'r') as txt:
        content = txt.read()

    # Open the C file in write mode and overwrite it with the content from the text file
    with open(os.path.join(save_dir,'main_prog_dot_c_copy.txt'), 'w') as cfile:
        cfile.write(content)

def extract_params(dir_path, save_dir = None):
    # Initialize an empty dictionary to store each parameter as a key with lists as values
    pattern_clk = r'\*addr = (0x[0-9A-Fa-f]+)'
    clk_mapping = {'0x81660019': 6940000, '0x81760019': 5950000, '0x81650019': 8330000} # hex PLL code to MLM clk in MHz (Ref SLATE-98)

    df_final = pd.DataFrame()
    df_final_end = pd.DataFrame()
    file = os.path.join(dir_path,'main_prog.c')

    # Read and parse the text file
    # file = r"C:\Users\AdrienPierre\Documents\slate_mpw3_slatekpi\bare_metal_c_interface\slate_c_mlm_finetuning\main_prog.c"
    with open(file, 'r') as file:
        current_struct = None  # Track the current struct for fine tune params
        current_struct_end = None # Track the current struct for endurance params
        current_struct_clk = None # Used for finding clk frequency

        for line in file:
            # Remove any comments after '//' and strip whitespace
            line = re.sub(r'//.*', '', line).strip()
            # print(line+'\n')

            # Skip empty lines
            if not line:
                continue

            ### Extract Fine Tuning Parameters ###
            # Check if we're starting a new struct
            struct_match = re.match(r'finetuning_para\s+(\w+)\s+=\s+{', line)
            if struct_match:
                # Create new dictionary with row name as group
                current_struct = struct_match.group(1)
                df_temp = pd.DataFrame({'wl_st': [0]}, index=[current_struct])
                continue

            # Check if we're at the end of a struct
            if (line == '};') and (current_struct != None):
                # Concat df_temp to final dataframe with label structure name as the index
                df_final = pd.concat([df_final, df_temp])

                current_struct = None
                continue

            # Extract the parameter name and value
            param_match = re.match(r'\.(\w+)\s*=\s*(\d+)', line)
            if bool(param_match) and (current_struct != None):
                df_temp[param_match.group(1)] = int(param_match.group(2))

            ### Endurance Parameters ###
            struct_match_end = re.match(r'endurance_para\s+(\w+)\s+=\s+{', line)
            if struct_match_end:
                # Create new dictionary with row name as group
                current_struct_end = struct_match_end.group(1)
                df_temp_end = pd.DataFrame({'wl_st': [0]}, index=[current_struct_end])
                continue

            # Check if we're at the end of a struct
            if (line == '};') and (current_struct_end != None):
                # Concat df_temp to final dataframe with label structure name as the index
                df_final_end = pd.concat([df_final_end, df_temp_end])

                current_struct_end = None
                continue

            # Extract the parameter name and value
            param_match_end = re.match(r'\.(\w+)\s*=\s*(\d+)', line)
            if bool(param_match_end) and (current_struct_end != None):
                df_temp_end[param_match.group(1)] = int(param_match_end.group(2))

            ### Find clock frequency ###
            struct_match_clk = re.match(r'void init_mlm\(\)\{', line)
            if struct_match_clk:
                current_struct_clk = True

            if current_struct_clk:
                param_match_clk = re.search(pattern_clk, line)
                if bool(param_match_clk):
                    clk_hex = param_match_clk.group(1)
                    clk_freq = clk_mapping[clk_hex]

    # Transpose, display and save the DataFrame
    time = '{:%Y-%m-%d %H-%M-%S}'.format(datetime.datetime.now())
    # path = Path(file)
    # dir = path.parent.absolute()
    print('Time now: ' + time + '\n')
    df_final = df_final.transpose()
    df_final_end = df_final_end.transpose()

    print('Fine Tuning Parameters\n')
    print(df_final)
    if save_dir == None:
        df_final.to_csv(os.path.join(dir_path,'Fine_Tune_Params.csv'))
    else:
        df_final.to_csv(os.path.join(save_dir,'Fine_Tune_Params.csv'))

    print('\nEndurance Parameters\n')
    print(df_final_end)
    if save_dir == None:
        df_final_end.to_csv(os.path.join(dir_path,'Endurance_Params.csv'))
    else:
        df_final_end.to_csv(os.path.join(save_dir,'Endurance_Params.csv'))

    # Convert outputs to analog values (voltage, time, current etc)
    df_final_analog = pd.DataFrame(columns = df_final.columns)
    for idx, row in df_final.iterrows():
        if ('_bl_' in idx) or ('_sl_' in idx): # Assumes AVDU28 at POR value of 2.8V
            row_analog = round(row * 0.0444444, 2)
            row_analog.name = row_analog.name + ' [V]'
            df_temp = pd.DataFrame(row_analog).transpose()
            df_final_analog = pd.concat([df_final_analog, df_temp])
        elif '_gate_' in idx:
            row_analog = round(row * 0.0018868 + 1.63, 2)
            row_analog.name = row_analog.name + ' [V]'
            df_temp = pd.DataFrame(row_analog).transpose()
            df_final_analog = pd.concat([df_final_analog, df_temp])
        elif ('_cycle_' in idx) and ('multi' not in idx):
            row_analog = round(row * (1/clk_freq), 9) # Cycle time is period of clk * steps
            row_analog.name = row_analog.name + ' [sec]'
            df_temp = pd.DataFrame(row_analog).transpose()
            df_final_analog = pd.concat([df_final_analog, df_temp])
    
    print('Analog Parameters\n')
    print(df_final_analog)
    if save_dir == None:
        df_final_analog.to_csv(os.path.join(dir_path,'Analog_Params.csv'))
    else:
        df_final_analog.to_csv(os.path.join(save_dir,'Analog_Params.csv'))

def flint_params_output_to_KPI_table(df, test_name_filter):
    # Find matching row
    matching_indices = df.index[df['TEST_NAME'] == test_name_filter].tolist()
    if len(matching_indices) == 1:
        row = matching_indices[0]
    elif (len(matching_indices) is None) or (len(matching_indices) == 0):
        raise Exception("No matching indices")
    elif len(matching_indices)>1:
        print("Choosing first occurence")
        row = matching_indices[0]

    # Select row
    df = df.iloc[row].to_frame()
    # df = df.loc[df.TEST_NAME==test_name_filter]

    # Filter out macro info but save in column name of data
    print("Converting ft params DB format to spreadsheet format")
    info_cols = ['RUN_NAME','TEST_START_DATETIME','STEPPING','DIE_ID','MACRO','IO','TEST_NAME','ALGO','CLK_MHZ','ADC_CAL'] # ,'BLREF_CAL'
    meta_data_list = [str(df.loc[x,row]) for x in info_cols]
    clk_freq = df.loc['CLK_MHZ',row] * 1e6
    data_col_name = '_'.join(meta_data_list)
    df.rename(columns={row: data_col_name}, inplace=True)
    df = df.iloc[df.index.isin(info_cols) == False]

    # Split by forming/ ft level
    df['Type'] = df.index.str.split('_').str[0]
    df['Param'] = df.index.str.split('_').str[1:].str.join('_')
    df = df.reset_index()
    # df = df.loc[df.Param.isin('ID')]
    df_piv = df.pivot(index=['Param'], columns=['Type'], values=[data_col_name]).reset_index()

    # Save
    dir_path = os.path.dirname(file)
    file_name = os.path.basename(file)
    ft_trim_save_path = os.path.join(dir_path, file_name.replace('.csv','_KPI_Tracker_Format.csv'))
    df_piv.to_csv(ft_trim_save_path, index=False)
    print(f'Saved here: {ft_trim_save_path}\n')

    # Convert outputs to analog values (voltage, time, current etc)
    df_piv.index = df_piv[df_piv.columns[0]]
    df_piv.drop(columns=[df_piv.columns[0]], inplace=True)
    df_final_analog = pd.DataFrame(columns = df_piv.columns)
    for idx, row in df_piv.iterrows():
        if ('_BL_' in idx) or ('_SL_' in idx): # Assumes AVDU28 at POR value of 2.8V
            row_analog = round(row * 0.042 + 0.035, 2)
            row_analog.name = row_analog.name + ' [V]'
            df_temp = pd.DataFrame(row_analog).transpose()
            df_final_analog = pd.concat([df_final_analog, df_temp])
        elif '_GATE_' in idx:
            row_analog = round(row * 0.00455 + 0.035, 2)
            row_analog.name = row_analog.name + ' [V]'
            df_temp = pd.DataFrame(row_analog).transpose()
            df_final_analog = pd.concat([df_final_analog, df_temp])
        elif ('_CYCLE_' in idx) and ('MULTI' not in idx):
            row_analog = round((row-4) * (1/clk_freq), 9) # Cycle time is period of clk * steps, subtract 2clk at beg and end for wl initialization
            row_analog.name = row_analog.name + ' [sec]'
            df_temp = pd.DataFrame(row_analog).transpose()
            df_final_analog = pd.concat([df_final_analog, df_temp])

    print('Calculate Analog Parameters')
    analog_save_path = os.path.join(dir_path, file_name.replace('.csv','_Analog_KPI_Tracker_Format.csv'))
    df_final_analog.to_csv(analog_save_path, index=True)

    print(f'Saved here: {analog_save_path}')

def extract_flint_params_from_c_code(dir_path, save_dir):
    # Initialize an empty dictionary to store each parameter as a key with lists as values
    df_final = pd.DataFrame()
    file = os.path.join(dir_path)

    # Read and parse the text file
    # file = r"C:\Users\AdrienPierre\Documents\slate_mpw3_slatekpi\bare_metal_c_interface\slate_c_mlm_finetuning\main_prog.c"
    with open(file, 'r') as file:
        current_struct_form = None  # Track the current struct for forming params
        current_struct = None  # Track the current struct for fine tune params
        current_struct_end = None # Track the current struct for endurance params

        for line in file:
            # Remove any comments after '//' and strip whitespace
            line = re.sub(r'//.*', '', line).strip()
            # print(line+'\n')

            # Skip empty lines
            if not line:
                continue

            ### Extract Fine Tuning Parameters ###
            # Check if we're starting a new struct
            struct_match = re.match(r'finetuning_para_t\s+(\w+)\s+=\s+{', line)
            if struct_match:
                # Create new dictionary with row name as group
                current_struct = struct_match.group(1)
                df_temp = pd.DataFrame({'wl_st': [0]}, index=[current_struct])
                continue

            # Check if we're at the end of a struct
            if (line == '};') and (current_struct != None):
                # Concat df_temp to final dataframe with label structure name as the index
                df_final = pd.concat([df_final, df_temp])

                current_struct = None
                continue

            # Extract the parameter name and value
            param_match = re.match(r'\.(\w+)\s*=\s*(\d+)', line)
            if bool(param_match) and (current_struct != None):
                df_temp[param_match.group(1)] = int(param_match.group(2))

    # Transpose, display and save the DataFrame
    time = '{:%Y-%m-%d %H-%M-%S}'.format(datetime.datetime.now())
    # path = Path(file)
    # dir = path.parent.absolute()
    print('Time now: ' + time + '\n')
    df_final = df_final.transpose()

    print('Fine Tuning Parameters\n')
    print(df_final)

    # Save fine tuning parameters
    for col in df_final.columns:
        df_param = {col: df_final[col].to_dict()}
        json_path = os.path.join(save_dir,col)

        # Save to a JSON file
        with open(json_path, 'w') as f:
            json.dump(df_param, f, indent=4)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert write params from DB format to excel format.")
    parser.add_argument(
        "file",
        type=str,
        help="Path to the input file (e.g., params_to_db.csv)",
    )

    # Parse arguments
    args = parser.parse_args()
    file = args.file

    # Verify the file exists
    if not os.path.isfile(file):
        print(f"Error: The specified file does not exist: {file}")
    else:
        print(f"Reading file: {file}")
        
        # Read the file into a DataFrame
        try:
            df = pd.read_csv(file, header=0)
            if 'TEST_NAME' not in df.columns:
                df = pd.read_csv(file, header=1)
            print("File loaded successfully.")
            # print(df.head())
        except Exception as e:
            print(f"Error reading file: {e}")
            exit(1)
        
        column_name = 'TEST_NAME'
        unique_values = list(df[column_name].unique())
        print(f"Unique values in column '{column_name}':")
        for idx, value in enumerate(unique_values):
            print(f"{idx}: {value}")
        
        # DEBUG COMMENT OUT
        selected_index = input(f"Enter the index of the {column_name} to select: ")
        
        flint_params_output_to_KPI_table(df, unique_values[int(selected_index)])
