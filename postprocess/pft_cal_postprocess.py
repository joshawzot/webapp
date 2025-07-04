# Post process parallel finetune window calibration data and save in test_calibration.json file
import pandas as pd
import json
import os

# Give list of level 1 std deviations
def post_process_pft_prewincal(die_id, macro, test_start_datetime, VCM_IDAC, VCM_RDAC, BL_R, postprocess_path):
    
    # Find the csv file and load it
    list_folders = os.listdir(postprocess_path)
    rwb_folder = [x for x in list_folders if test_start_datetime in x]
    if len(rwb_folder)==1:
        rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from tester.csv')
    else:
        raise Exception('Multiple post process files with the same test start date and time found')
    df = pd.read_csv(rwb_file)

    # Find the current readout
    df = df.loc[df.TEST_NAME.str.contains(f'i{VCM_IDAC}r{VCM_RDAC}b{BL_R}-level1')]

    level1_distsigma_list = df.LEVEL_1_DISTSIGMA.to_list()
    level1_median_list = df.LEVEL_12_1_0SIGMA_ADC.to_list()

    return level1_distsigma_list, level1_median_list

def post_process_pft_prewincal_tracking(die_id, macro, test_start_datetime, TRACK_VCM_ROUGH, TRACK_BLDRV_BL_R, postprocess_path):
    
    # Find the csv file and load it
    list_folders = os.listdir(postprocess_path)
    rwb_folder = [x for x in list_folders if test_start_datetime in x]
    if len(rwb_folder)==1:
        rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from tester.csv')
    else:
        raise Exception('Multiple post process files with the same test start date and time found')
    df = pd.read_csv(rwb_file)

    # Find the current readout
    df = df.loc[df.TEST_NAME.str.contains(f'trackR{TRACK_VCM_ROUGH}b{TRACK_BLDRV_BL_R}-level1')]

    level1_distsigma_list = df.LEVEL_1_DISTSIGMA.to_list()
    level1_median_list = df.LEVEL_12_1_0SIGMA_ADC.to_list()

    return level1_distsigma_list, level1_median_list


def post_process_pft_wincal(die_id, macro, test_start_datetime, lvl1_target, BL_R, mode, wc_iter, postprocess_path, error_tol = 0.02):
    
    # Find the csv file and load it
    list_folders = os.listdir(postprocess_path)
    rwb_folder = [x for x in list_folders if test_start_datetime in x]
    if len(rwb_folder)==1:
        rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from_tester.csv')
    else:
        raise Exception('Multiple post process files with the same test start date and time found')
    df = pd.read_csv(rwb_file)

    # Filter to wincal data and for each IO find the TRACK_VCM_ROUGH value that gets level1 std dev closest to its target
    if mode == 'vcm':
        df = df.loc[df.TEST_NAME.str.contains('wcvcmrough')]
        df['TRACK_VCM_ROUGH'] = df.TEST_NAME.str.extract(r'wcvcmrough(\d+)-level1').astype(int)
    elif mode == 'bleeder': # Treat bleeder as vcm but with TRACK_VCM_ROUGH values decreasing monotonically
        df = df.loc[df.TEST_NAME.str.contains('wcbleeder')]
        df['TRACK_VCM_ROUGH'] = 31-df.TEST_NAME.str.extract(r'wcbleeder(\d+)-level1').astype(int)
    df['BL_R'] = BL_R

    df['LEVEL_1_DISTSIGMA - lvl1target'] = df.LEVEL_1_DISTSIGMA - lvl1_target
    df['abs(LEVEL_1_DISTSIGMA - lvl1target)'] = abs(df.LEVEL_1_DISTSIGMA - lvl1_target)
    df['min(LEVEL_1_DISTSIGMA)'] = df.groupby('IO').LEVEL_1_DISTSIGMA.transform('min')

    # # Save optimal results in their own dataframe [IN PREVIOUS VERSION]
    # df_opt_vcmrough = df.loc[df.groupby('IO')['abs(LEVEL_1_DISTSIGMA - lvl1target)'].idxmin()]

    # Save most recent results (assumes TRACK_VCM_ROUGH increasing monotically) [NEW CODE]
    df_opt_vcmrough = df.loc[df.groupby('IO')['TRACK_VCM_ROUGH'].idxmax()]
    
    # Mark IOs that land within the error tolerate as meeting the target
    df_opt_vcmrough['Meets target'] = df_opt_vcmrough.apply(lambda x: 'Y' if x['abs(LEVEL_1_DISTSIGMA - lvl1target)']<=error_tol else 'N', axis=1)
    
    # Mark IOs that are sampled both below and above sigma target as meeting the target
    df_min_max = df.groupby('IO')['LEVEL_1_DISTSIGMA - lvl1target'].agg(['min','max']).reset_index()
    df_min_max['Meets target'] = df_min_max.apply(lambda x: 'Y' if ((x.min() < 0) and (x.max() > 0)) else 'N', axis=1)
    
    # If level1 std dev is starting to worsen beyond a threshold, stop testing for that IO
    worsening_threshold = 0.1
    df_opt_vcmrough['Worsening'] = df_opt_vcmrough.apply(lambda x: 'Y' if x['LEVEL_1_DISTSIGMA'] > (x['min(LEVEL_1_DISTSIGMA)'] + worsening_threshold) else 'N', axis=1)
    for idx, row in df_opt_vcmrough.iterrows():
        if row.Worsening == 'Y':
            print(f' - - IO{row.IO} level1 std dev is worsening beyond threshold of {worsening_threshold}, will stop wincal on this IO')
    df_opt_vcmrough['Meets target'] = df_opt_vcmrough.apply(lambda x: 'Y' if x.Worsening=='Y' else x['Meets target'], axis=1)
    
    # Merge the two dataframes to get a unified list of IOs that meet or don't meet the target criteria 
    df_min_max = df_min_max.loc[df_min_max['Meets target']=='Y'].set_index('IO')
    df_opt_vcmrough = df_opt_vcmrough.set_index('IO')
    df_opt_vcmrough.update(df_min_max)
    df_opt_vcmrough.reset_index(inplace=True)
    
    # Clearly define IOs that meet and don't meet target in separate dataframes
    df_converged = df_opt_vcmrough.loc[df_opt_vcmrough['Meets target']=='Y']
    df_non_converged = df_opt_vcmrough.loc[df_opt_vcmrough['Meets target']=='N']

    # Save IOs that meets target to json file
    root_dir = os.path.dirname(os.path.dirname(postprocess_path))
    json_file = os.path.join(root_dir, f'{die_id.lower()}_macro{macro}_calibration_result.json')

    with open(json_file, "r") as file:
        data = json.load(file)

    for idx, row in df_converged.iterrows():
        if mode == 'vcm':
            if row.Worsening == 'N':
                # data[f'WINCAL_VCM_IDAC_io{row.IO}'] = row.VCM_IDAC
                data[f'WINCAL_TRACK_VCM_ROUGH_io{row.IO}'] = row.TRACK_VCM_ROUGH
                data[f'WINCAL_BL_R_io{row.IO}'] = row.BL_R
            else: # Use VCM_IDAC value that gives lowest level1 std dev
                # optimal_vcm_idac = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].VCM_IDAC.values[0]
                optimal_vcm_rough = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].TRACK_VCM_ROUGH.values[0]
                optimal_bl_r = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].BL_R.values[0]

                # data[f'WINCAL_VCM_IDAC_io{row.IO}'] = int(optimal_vcm_idac)
                data[f'WINCAL_VCM_RDAC_io{row.IO}'] = int(optimal_vcm_rough)
                data[f'WINCAL_BL_R_io{row.IO}'] = int(optimal_bl_r)
        elif mode == 'bleeder':
            if row.Worsening == 'N':
                # data[f'WINCAL_VCM_IDAC_io{row.IO}'] = row.VCM_IDAC
                data[f'WINCAL_BLEED_RD_io{row.IO}'] = 31 - row.TRACK_VCM_ROUGH
                data[f'WINCAL_BL_R_io{row.IO}'] = row.BL_R
            else: # Use VCM_IDAC value that gives lowest level1 std dev
                # optimal_vcm_idac = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].VCM_IDAC.values[0]
                optimal_vcm_rough = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].TRACK_VCM_ROUGH.values[0]
                optimal_bl_r = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].BL_R.values[0]

                # data[f'WINCAL_VCM_IDAC_io{row.IO}'] = int(optimal_vcm_idac)
                data[f'WINCAL_BLEED_RD_io{row.IO}'] = 31 - int(optimal_vcm_rough)
                data[f'WINCAL_BL_R_io{row.IO}'] = int(optimal_bl_r)

    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)  # Save with indentation for readability

    print(' - JSON calibration file updated')

    # Return IOs that haven't met the target +- error tolerance
    nonconverged_ios = [int(x) for x in df_non_converged.IO.unique()]

    if len(nonconverged_ios)>0:
        print(f' - {len(nonconverged_ios)} remaining IOs to calibrate for level 1 std deviation:')
        print(f'   {nonconverged_ios}')
        print(f'   Level1 mean std deviation {df_non_converged.LEVEL_1_DISTSIGMA.mean():.2f} vs. target of {lvl1_target}')

    # Raise a warning if there is an IO where the minimum VCM setting is below the std deviation target
    df_min_vcmrough = df.loc[df.groupby('IO')['TRACK_VCM_ROUGH'].idxmin()].reset_index()
    df_min_vcmrough = df_min_vcmrough.loc[df_min_vcmrough.LEVEL_1_DISTSIGMA < (lvl1_target - error_tol)]

    # if df_min_vcmidac.empty == False:
    #     print('\n*******************')
    #     print(f' - The following IOs are already below the level1 std dev target at the minimum VCM_IDAC setting:\n{list(df_min_vcmidac.IO.unique())}')
    #     print('   Resulting conductance window for these IOs will likely be too high, suggest starting window calibration with a lower initial setting')
    #     print('*******************\n')

    return nonconverged_ios


def post_process_pft_wincal_tracking(die_id, macro, test_start_datetime, lvl1_target, TRACK_BLDRV_BL_R, postprocess_path, error_tol = 0.02):
    
    # Find the csv file and load it
    list_folders = os.listdir(postprocess_path)
    rwb_folder = [x for x in list_folders if test_start_datetime in x]
    if len(rwb_folder)==1:
        rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from tester.csv')
    else:
        raise Exception('Multiple post process files with the same test start date and time found')
    df = pd.read_csv(rwb_file)

    # Filter to wincal data
    df = df.loc[df.TEST_NAME.str.contains('wctrackR')]

    # For each IO find the TRACK_VCM_ROUGH value that gets level1 std dev closest to its target
    df['TRACK_VCM_ROUGH'] = df.TEST_NAME.str.extract(r'wctrackR(\d+)-level1').astype(int)
    df['TRACK_BLDRV_BL_R'] = TRACK_BLDRV_BL_R

    df['LEVEL_1_DISTSIGMA - lvl1target'] = df.LEVEL_1_DISTSIGMA - lvl1_target
    df['abs(LEVEL_1_DISTSIGMA - lvl1target)'] = abs(df.LEVEL_1_DISTSIGMA - lvl1_target)
    df['min(LEVEL_1_DISTSIGMA)'] = df.groupby('IO').LEVEL_1_DISTSIGMA.transform('min')

    # # Save optimal results in their own dataframe [IN PREVIOUS VERSION]
    # df_opt_vcmrough = df.loc[df.groupby('IO')['abs(LEVEL_1_DISTSIGMA - lvl1target)'].idxmin()]

    # Save most recent results (assumes TRACK_VCM_ROUGH increasing monotically) [NEW CODE]
    df_opt_trackR = df.loc[df.groupby('IO')['TRACK_VCM_ROUGH'].idxmax()]
    
    # Mark IOs that land within the error tolerate as meeting the target
    df_opt_trackR['Meets target'] = df_opt_trackR.apply(lambda x: 'Y' if x['abs(LEVEL_1_DISTSIGMA - lvl1target)']<=error_tol else 'N', axis=1)
    
    # Mark IOs that are sampled both below and above sigma target as meeting the target
    df_min_max = df.groupby('IO')['LEVEL_1_DISTSIGMA - lvl1target'].agg(['min','max']).reset_index()
    df_min_max['Meets target'] = df_min_max.apply(lambda x: 'Y' if ((x.min() < 0) and (x.max() > 0)) else 'N', axis=1)
    
    # If level1 std dev is starting to worsen beyond a threshold, stop testing for that IO
    worsening_threshold = 0.1
    df_opt_trackR['Worsening'] = df_opt_trackR.apply(lambda x: 'Y' if x['LEVEL_1_DISTSIGMA'] > (x['min(LEVEL_1_DISTSIGMA)'] + worsening_threshold) else 'N', axis=1)
    for idx, row in df_opt_trackR.iterrows():
        if row.Worsening == 'Y':
            print(f' - - IO{row.IO} level1 std dev is worsening beyond threshold of {worsening_threshold}, will stop wincal on this IO')
    df_opt_trackR['Meets target'] = df_opt_trackR.apply(lambda x: 'Y' if x.Worsening=='Y' else x['Meets target'], axis=1)
    
    # Merge the two dataframes to get a unified list of IOs that meet or don't meet the target criteria 
    df_min_max = df_min_max.loc[df_min_max['Meets target']=='Y'].set_index('IO')
    df_opt_trackR = df_opt_trackR.set_index('IO')
    df_opt_trackR.update(df_min_max)
    df_opt_trackR.reset_index(inplace=True)
    
    # Clearly define IOs that meet and don't meet target in separate dataframes
    df_converged = df_opt_trackR.loc[df_opt_trackR['Meets target']=='Y']
    df_non_converged = df_opt_trackR.loc[df_opt_trackR['Meets target']=='N']

    # Save IOs that meets target to json file
    root_dir = os.path.dirname(os.path.dirname(postprocess_path))
    json_file = os.path.join(root_dir, f'{die_id.lower()}_macro{macro}_calibration_result.json')

    with open(json_file, "r") as file:
        data = json.load(file)

    for idx, row in df_converged.iterrows():
        if row.Worsening == 'N':
            data[f'WINCAL_TRACK_VCM_ROUGH_io{row.IO}'] = row.TRACK_VCM_ROUGH
            data[f'WINCAL_TRACK_BL_R_io{row.IO}'] = row.TRACK_BLDRV_BL_R
        else: # Use VCM_IDAC value that gives lowest level1 std dev
            optimal_trackR = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].VCM_IDAC.values[0]
            optimal_bl_r = df.loc[(df['min(LEVEL_1_DISTSIGMA)']==df['LEVEL_1_DISTSIGMA'])&(df.IO==row.IO)].BL_R.values[0]

            data[f'WINCAL_TRACK_VCM_ROUGH_io{row.IO}'] = int(optimal_trackR)
            data[f'WINCAL_TRACK_BL_R_io{row.IO}'] = int(optimal_bl_r)

    with open(json_file, "w") as file:
        json.dump(data, file, indent=4)  # Save with indentation for readability

    print(' - JSON calibration file updated')

    # Return IOs that haven't met the target +- error tolerance
    nonconverged_ios = [int(x) for x in df_non_converged.IO.unique()]

    if len(nonconverged_ios)>0:
        print(f' - {len(nonconverged_ios)} remaining IOs to calibrate for level 1 std deviation:')
        print(f'   {nonconverged_ios}')
        print(f'   Level1 mean std deviation {df_non_converged.LEVEL_1_DISTSIGMA.mean():.2f} vs. target of {lvl1_target}')

    # Raise a warning if there is an IO where the minimum VCM setting is below the std deviation target
    df_min_trackR = df.loc[df.groupby('IO')['TRACK_VCM_ROUGH'].idxmin()].reset_index()
    df_min_trackR = df_min_trackR.loc[df_min_trackR.LEVEL_1_DISTSIGMA < (lvl1_target - error_tol)]

    # if df_min_vcmidac.empty == False:
    #     print('\n*******************')
    #     print(f' - The following IOs are already below the level1 std dev target at the minimum VCM_IDAC setting:\n{list(df_min_vcmidac.IO.unique())}')
    #     print('   Resulting conductance window for these IOs will likely be too high, suggest starting window calibration with a lower initial setting')
    #     print('*******************\n')

    return nonconverged_ios

def post_process_pft_ronscal(die_id, macro, test_start_datetime, ppm_target, postprocess_path, VCM_IDAC, VCM_RDAC, BL_R, target_col_name):
    
    # Find the csv file and load it
    list_folders = os.listdir(postprocess_path)
    rwb_folder = [x for x in list_folders if test_start_datetime in x]
    if len(rwb_folder)==1:
        rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from tester.csv')
    else:
        raise Exception('Multiple post process files with the same test start date and time found')
    df = pd.read_csv(rwb_file)

    # Filter to most recent readout
    df = df.loc[df.TEST_NAME.str.contains(f'wcvcmidac{VCM_IDAC}-level0')]

    # Calculate mean RonS PPM
    df['RonS PPM'] = 1e6 - df[target_col_name]

    mean_rons_ppm = df['RonS PPM'].mean()
    if mean_rons_ppm > ppm_target:
        print(f' - - Reset-on-set PPM of {mean_rons_ppm} is above threshold of {ppm_target}')

        # Save optimal calbit values
        optimal_vcm_idac = VCM_IDAC - 1
        optimal_vcm_rdac = VCM_RDAC
        optimal_bl_r = BL_R
        print(f' - - Saving optimal calbit settings of:\n - - - VCM_IDAC: {optimal_vcm_idac}')

        root_dir = os.path.dirname(os.path.dirname(postprocess_path))
        json_file = os.path.join(root_dir, f'{die_id.lower()}_macro{macro}_calibration_result.json')

        with open(json_file, "r") as file:
            data = json.load(file)

        data['VCM_IDAC_pft'] = optimal_vcm_idac
        data['VCM_RDAC_pft'] = optimal_vcm_rdac
        data['BLDRV_BL_R_pft'] = optimal_bl_r

        with open(json_file, "w") as file:
            json.dump(data, file, indent=4)  # Save with indentation for readability

        print(' - JSON calibration file updated')

        # Stop testing
        cont = False
    else:
        # Continue testing
        print(f' - - Reset-on-set PPM of {mean_rons_ppm} below threshold of {ppm_target}')
        cont = True

    return cont

# def post_process_pft_clipping(die_id, macro, test_start_datetime, ppm_target, postprocess_path, VCM_IDAC, VCM_RDAC, BL_R, target_col_name='LEVEL_3_MAXADC'):
    
#     # Find the csv file and load it
#     list_folders = os.listdir(postprocess_path)
#     rwb_folder = [x for x in list_folders if test_start_datetime in x]
#     if len(rwb_folder)==1:
#         rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from tester.csv')
#     else:
#         raise Exception('Multiple post process files with the same test start date and time found')
#     df = pd.read_csv(rwb_file)

#     # Filter to most recent readout
#     df = df.loc[df.TEST_NAME.str.contains(f'wcvcmidac{VCM_IDAC}-level3')]

#     # Calculate mean RonS PPM
#     min_of_maxadc = df[target_col_name].min()

#     if min_of_maxadc < 63:
#         print(f' - - Detect clipping with min(max(level3)) of {min_of_maxadc}')

#         # Save optimal calbit values
#         optimal_vcm_idac = VCM_IDAC - 1
#         optimal_vcm_rdac = VCM_RDAC
#         optimal_bl_r = BL_R
#         print(f' - - Saving optimal calbit settings of:\n - - - VCM_IDAC: {optimal_vcm_idac}')

#         root_dir = os.path.dirname(os.path.dirname(postprocess_path))
#         json_file = os.path.join(root_dir, f'{die_id.lower()}_macro{macro}_calibration_result.json')

#         with open(json_file, "r") as file:
#             data = json.load(file)

#         data['VCM_IDAC_pft'] = optimal_vcm_idac
#         data['VCM_RDAC_pft'] = optimal_vcm_rdac
#         data['BLDRV_BL_R_pft'] = optimal_bl_r

#         with open(json_file, "w") as file:
#             json.dump(data, file, indent=4)  # Save with indentation for readability

#         print(' - JSON calibration file updated')

#         # Stop testing
#         cont = False
#     else:
#         # Continue testing
#         print(f' - - No ADC clipping detected')
#         cont = True

#     return cont

def post_process_pft_clipping(die_id, macro, test_start_datetime, ppm_target, postprocess_path, VCM_IDAC, VCM_RDAC, BL_R, target_col_name='LEVEL_3_MAXADC'):
    
    # Find the csv file and load it
    list_folders = os.listdir(postprocess_path)
    rwb_folder = [x for x in list_folders if test_start_datetime in x]
    if len(rwb_folder)==1:
        rwb_file = os.path.join(postprocess_path, rwb_folder[0], 'rwb_from tester.csv')
    else:
        raise Exception('Multiple post process files with the same test start date and time found')
    df = pd.read_csv(rwb_file)

    # Filter to most recent readout
    df = df.loc[df.TEST_NAME.str.contains('ADCclipping-level3')]

    # Return list of max ADC for each IO
    list_max_adc = df[target_col_name].to_list()

    return list_max_adc

def post_process_pft_adcoffsetwincal(die_id, macro, test_start_datetime, ppm_target, postprocess_path, VCM_IDAC, VCM_RDAC, BL_R, target_col_name='LEVEL_3_MAXADC'):
    print('In ADC offset wincal')


if __name__ == "__main__":
    ios = post_process_pft_wincal('TT16', 1, '2025_04_01-09_49_08', 1.75, 11, 12)
    print(ios)