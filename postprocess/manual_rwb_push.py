# Script to push RWB data manually to server if there is an issue
# import numpy as np
# import time
import os
# import re
import sys
# import trio
# import pytest
# import math
# import git
# import json
# from pathlib import Path
# import matplotlib.pyplot as plt
# import time
# import datetime
import pandas as pd
# import test_conductance_range as tcr
# import scipy.stats as stats
# import bisect

# # Determine absolute directory paths
# automation_file_path = os.path.abspath(__file__)
# devtests_dir = os.path.dirname(automation_file_path)
# flint_dir = os.path.dirname(devtests_dir)
# tests_dir = os.path.dirname(flint_dir)
# repo_dir = os.path.dirname(tests_dir)
# postprocess_dir = os.path.join(tests_dir, "postprocess")

# # Import python files from postprocess folder
# sys.path.insert(1, postprocess_dir)
# import core_post_processing_functions as cf
# import convert_form_precycle_finetune_param_to_table as ft_table
# import create_and_upload as cu
import create_and_upload_rwb_param as curp

# Database index
rwb_db_index = 3

# Define post process folder from which to fetch rwb_from tester.csv files
folders_list = [r'C:/Users/AdrienPierre/Documents/debug/TO upload/2025_01_28-18_13_33_TST_flint_tt24_macro2',
                r'C:/Users/AdrienPierre/Documents/debug/TO upload/2025_01_29-06_41_07_TST_flint_tt24_macro2',
                r'C:/Users/AdrienPierre/Documents/debug/TO upload/2025_01_29-06_47_04_TST_flint_tt24_macro2',
                r'C:/Users/AdrienPierre/Documents/debug/TO upload/2025_01_29-18_09_53_TST_flint_tt24_macro2',
                r'C:/Users/AdrienPierre/Documents/debug/TO upload/2025_01_30-17_32_56_TST_flint_tt24_macro2']

rwb = pd.DataFrame()
for folder_path in folders_list:
    # Load RWB file
    rwb_file_path = os.path.join(folder_path, 'rwb_from tester.csv')
    rwb = pd.concat([rwb, pd.read_csv(rwb_file_path)], ignore_index=True)

print('Pushing concatenated RWB file')
curp.upload_to_db(rwb, table_name=f'rwb_db_{rwb_db_index}', DATABASE_NAME='rwb')