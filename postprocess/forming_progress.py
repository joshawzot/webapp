# Import forming progress
import pandas as pd
import numpy as np
import sys, os, re, math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy.stats as stats
#sys.path.insert(1, r"C:\Users\AdrienPierre\Documents\flint_mpw4_testing\tests\postprocess")
sys.path.insert(1, '/home/admin2/webapp_2/postprocess')
import core_post_processing_functions as cf
import create_and_upload_rwb_param as curp

def convert_to_range_string(lst):
    """Convert a list of numbers into a range string format."""
    lst.sort()  # Ensure the list is sorted
    ranges = []
    start = lst[0]

    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1] + 1:  # Check if the sequence is broken
            end = lst[i - 1]
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = lst[i]

    # Append the last range
    ranges.append(f"{start}-{lst[-1]}" if start != lst[-1] else f"{start}")

    return ", ".join(ranges)

# Pull data
rwb_pull = cf.rwb_fetch_data(regex='form', regex_col='TEST_NAME')

# Restrict to dies of interest
# die_id = ['TT21','TT22','TT24','TT25']
die_macro_filter = pd.DataFrame({'DIE_ID': ['TT21','TT22','TT25','TT21','TT22','TT24','TT25', 'TT24'],
                                 'MACRO': [2, 2, 2, 3, 3, 3, 3, 4]})

# print(rwb_pull.loc[rwb_pull.DIE_ID=='tt24'].RUN_NAME.unique())

# macro = 2
test_contains = 'form' # only looks for readouts with 'form' in the name

# rwb_formed = rwb_pull.loc[(rwb_pull.DIE_ID.isin(die_id))&(rwb_pull.MACRO==macro)&(rwb_pull.TEST_NAME.str.contains('form'))][['DIE_ID','MACRO','IO']].drop_duplicates()
rwb_formed = rwb_pull.loc[(rwb_pull.TEST_NAME.str.contains('form'))][['DIE_ID','MACRO','IO']].drop_duplicates()
rwb_formed = pd.merge(left=rwb_formed, right=die_macro_filter, how='inner')

rwb_formed_list = rwb_formed.groupby(['DIE_ID','MACRO'])['IO'].agg(list).reset_index()
rwb_formed_list['IO'] = rwb_formed_list['IO'].apply(convert_to_range_string)

print(rwb_formed_list)