# Script to push RWB data manually to server if there is an issue
import os
import pandas as pd
import create_and_upload_rwb_param as curp

# Database index
rwb_db_index = 3

# # Define post process folder from which to fetch rwb_from tester.csv files
# root_dir = r'C:\Users\AdrienPierre\Documents\F2MT-327_Rev7KPIrnd2\RWB_files'
# folders_list = os.listdir(root_dir)

# rwb = pd.DataFrame()
# for folder in folders_list:
#     for file in os.listdir(os.path.join(root_dir, folder)):
#         # Load RWB file
#         if file =='rwb_from tester.csv':
#             rwb_file_path = os.path.join(root_dir, folder, file)
#             rwb = pd.concat([rwb, pd.read_csv(rwb_file_path)], ignore_index=True)

# Load individual file
rwb = pd.read_csv(r"C:\Users\AdrienPierre\Documents\F2MT-361_PFTat3p3V\rwb_from tester bl38to63 100cyc two step ft.csv")

# Don't uplaod wincal data
rwb = rwb.loc[~rwb.TEST_NAME.str.contains('wincal')]

print('Pushing concatenated RWB file')
curp.upload_to_db(rwb, table_name=f'rwb_db_{rwb_db_index}', DATABASE_NAME='rwb')