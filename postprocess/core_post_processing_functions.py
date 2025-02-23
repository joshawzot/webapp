# Core post processing functions
import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats
import bisect
from itertools import chain
import math
import matplotlib.colors as mcolors
import scipy.interpolate as spi
import paramiko
import random
import time

# ### USER INPUTS ###
# # pattern choices: row_dir_bar, checkerboard
# pattern = 'rowbar'
# ###################

# if pattern == 'rowbar':
#     target_pattern = np.zeros((64,1296))
#     target_pattern[:,324:324*2] = 1
#     target_pattern[:,324*2:324*3] = 2
#     target_pattern[:,324*3:] = 3

def plot_nq(data_arrays, ):
    plt.figure(figsize=(8, 6))
    # Generate a color map to differentiate each dataset visually
    colors = ['r', 'b', 'g', 'm']

    for idx, data in enumerate(data_arrays):
        # Sort the data and calculate quantiles
        sorted_data = np.sort(data)
        quantiles = np.linspace(0, 1, len(sorted_data))

        # Transform quantiles into standard normal variates
        normal_quantiles = stats.norm.ppf(quantiles)

        # Plot with a normal distribution scale on the y-axis
        plt.plot(sorted_data, normal_quantiles, marker='o', linestyle='-', color=colors[idx], label=f'Data {idx+1}')

    # Add labels and legend
    plt.xlabel('Data Values')
    plt.ylabel('Normal Quantiles')
    plt.title('Overlay of Data Distributions on a Normal Scale')
    plt.legend()
    plt.grid(True)
    plt.show()

# Data input is passed as a dictionary and titles are the order of the keys desired to be plotted
def plot_bit_fail_map(data, titles = ['plot 1','plot 2','plot 3','plot 4']):
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes_flat = axes.flatten()
    if len(titles) <= 4:
        for i, plot_name in enumerate(titles):
            plot_data = data[plot_name]
            axes_flat[i].imshow(plot_data, cmap='viridis', interpolation = None, vmin=plot_data.min(), vmax=plot_data.max(), aspect='auto')
            axes_flat[i].set_title(plot_name, fontsize=24)
            axes_flat[i].set_xlabel('Row (WL)')
            axes_flat[i].set_ylabel('Col (BL)')

        for i in range(len(titles), len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
    else:
        print('Too many fails maps, must be less than or equal to 4')

# Function to generate subplots of NQ distributions, data_list is a dictionary with the first level the readout name and secondary level the bit pattern (aka level).
# Titles is the keys of data in the order they wish to be plotted.
# Returns a dictionary of read window budget (rwb) for each key in the data plotted and the interpolated ppm distributions
# cpm_range is a tuple of (adc_code_0_cond, adc_code_63_cond)
def rwb_intcond_overlaylevelnqplot(data_list, titles = ['plot 1','plot 2','plot 3','plot 4'], ref_ppm_list = [170,500], xpoint_levels_list = [(0,1),(1,2),(2,3)], 
                                   enable_plots = False, plot_ref_lines = True, cond_range=(30,160)):
    if enable_plots:
        if len(titles) > 4:
            num_rows = math.ceil(len(titles)/2)
            fig, axs = plt.subplots(num_rows, 2, figsize=(14, 2+num_rows*5.7))  # 2 row, 2 columns of subplots
            axs = axs.flatten()
        elif len(titles) > 2:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 row, 2 columns of subplots
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 row, 2 columns of subplots
            axs = axs.flatten()

        # Colors for each line in the plots
        colors = ['r', 'b', 'g', 'm']

    # Reference ppm list is 0, -1, -2, -2.5, -3 sigma and whatever inputs passed for plot referencing
    sigma_list = [0, -1, -2, -2.5, -3]
    ref_ppmsigma_list = [('sigma', x) for x in sigma_list]
    ref_ppmsigma_list = ref_ppmsigma_list + [('ppm', x) for x in ref_ppm_list]

    # Convert sigma list into a string for column names
    str_sigma_list = [str(x).replace('.','P') for x in sigma_list]
    str_sigma_list = [x.replace('-','M') for x in str_sigma_list]

    # Initialize rwb dataframe
    # min_cond = 100 # in uS
    # max_cond = 0 # in uS
    # for key in data_list.keys():
    #     for level in data_list[key].keys():
    #         min_cond_temp = np.min(data_list[key][level])
    #         max_cond_temp = np.max(data_list[key][level])
    #         if min_cond_temp < min_cond:
    #             min_cond = min_cond_temp
    #         if max_cond_temp > max_cond:
    #             max_cond = max_cond_temp
    # interpolated_cond_range = np.arange(int(min_cond/2)*2, int(max_cond/2)*2+2, 2)
    rwb = pd.DataFrame(np.nan, index=titles, columns=[['LEVEL_%d%d_XPOINT_PPM' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        ['LEVEL_%d%d_XPOINT_ADC' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        ['LEVEL_%d%d_XPOINT_SUM_PPM' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        ['LEVEL_%d%d_XPOINT_SUM_ADC' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        list(chain.from_iterable([['LEVEL_%d%d_%sSIGMA_ADCRWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([['LEVEL_%d%d_%dPPM_ADCRWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in ref_ppm_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%dPPM_ADC' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%dPPM_ADC' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in ref_ppm_list])) + 
                        list(chain.from_iterable([['LEVEL_%d_MAXADC' % level, 'LEVEL_%d_MINADC' % level] for level in [0,1,2,3]])) + 
                        ['LEVEL_%d_DISTSIGMA' % level for level in [0,1,2,3]] + 
                        ['ADC_0_COND_US','ADC_63_COND_US'] + 
                        ['LEVEL_0_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_1_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_2_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_3_%dADC_PPM' % cond for cond in range(64)], 
                        # ['LEVEL_01_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_12_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_23_%dADC_PPM' % cond for cond in range(64)],
                        ])
    
    for i, key in enumerate(titles):
        # First subplot for `rowbar_1d`
        j = 0
        sorted_data = [np.nan, np.nan, np.nan, np.nan]
        normal_quantiles = [np.nan, np.nan, np.nan, np.nan]
        for level, plot_data in data_list[key].items():
            sorted_data[j] = np.sort(plot_data)
            quantiles = np.linspace(0, 1, len(sorted_data[j]))
            normal_quantiles[j] = stats.norm.ppf(quantiles)
            if enable_plots:
                axs[i].plot(sorted_data[j], normal_quantiles[j], marker='.', linestyle='-', color=colors[j], label='LEVEL %d' % level)
            j += 1

        if enable_plots:
            # Draw reference lines
            x_min, x_max = axs[i].get_xlim()
            x_pos = (x_min + x_max)/2 # draw at mid point
            if plot_ref_lines:
                for ppm in ref_ppm_list:
                    ref_sigma = stats.norm.ppf(ppm/1e6)
                    
                    axs[i].axhline(y=ref_sigma, color='gray', linestyle='--')
                    axs[i].text(x=x_pos, y=ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label
                    
                    axs[i].axhline(y=-ref_sigma, color='gray', linestyle='--')
                    axs[i].text(x=x_pos, y=-ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label

            # Customize first subplot
            axs[i].set_xlabel('Conductance [uS]')
            axs[i].set_ylabel('Normal Quantiles')
            axs[i].set_title('Overlay of Distributions for %s' % key)
            axs[i].legend() # [ 'Level %d' % x for x in range(4)]
            axs[i].grid(True)

        # if output_nq_data:
        #     # Convert 
        #     nq_output = 

        ### Calculate RWB ###
        # Save conductance range for later post processing
        rwb.loc[key,'ADC_0_COND_US'] = cond_range[0]
        rwb.loc[key,'ADC_63_COND_US'] = cond_range[1]
        # xpoint_levels_list = [(0,1),(1,2),(2,3),(0,3)] # List of intersection levels at which to calculate rwb
        for lower_level, upper_level in xpoint_levels_list:
            for stat_type, stat_value in ref_ppmsigma_list:
                # Convert ppm into sigma
                if stat_type == 'ppm':
                    ref_sigma = stats.norm.ppf(stat_value/1e6)
                elif stat_type =='sigma':
                    ref_sigma = stat_value

                # Find conductance at ppm for lower level, sigma is positive
                index_lower_level = bisect.bisect_right(normal_quantiles[lower_level], abs(ref_sigma))
                cond_lower_level = sorted_data[lower_level][index_lower_level]

                # Find conductance at ppm for upper level, sigma is negative
                index_upper_level = bisect.bisect_right(normal_quantiles[upper_level], -abs(ref_sigma))
                cond_upper_level = sorted_data[upper_level][index_upper_level]

                # Save RWB and lower and upper conductances for this particular test and xpoint
                if stat_type == 'ppm':
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_ADC' % (lower_level, upper_level, lower_level, round(stat_value))] = cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_ADC' % (lower_level, upper_level, upper_level, round(stat_value))] = cond_upper_level
                    rwb.loc[key,'LEVEL_%d%d_%dPPM_ADCRWB' % (lower_level, upper_level, round(stat_value))] = cond_upper_level - cond_lower_level
                elif stat_type =='sigma':
                    print_sigma = str(ref_sigma).replace('.','P') # replace period with 'P'
                    print_sigma = print_sigma.replace('-','M') # replace minus sign with 'M'
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower_level, upper_level, lower_level, print_sigma)] = cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower_level, upper_level, upper_level, print_sigma)] = cond_upper_level
                    rwb.loc[key,'LEVEL_%d%d_%sSIGMA_ADCRWB' % (lower_level, upper_level, print_sigma)] = cond_upper_level - cond_lower_level

            # Calculate xpoint adc and ppm
            # Sum ppm from lower and upper level at matched conductance
            df_lower = pd.DataFrame({'ADC': sorted_data[lower_level], 
                                            'ppm': stats.norm.cdf(-normal_quantiles[lower_level])*1e6})
            df_upper = pd.DataFrame({'ADC': sorted_data[upper_level], 
                                            'ppm': stats.norm.cdf(normal_quantiles[upper_level])*1e6})
            
            # Find highest ppm for each ADC value then combine into a dataframe
            df_lower_uniquecond = df_lower.groupby(['ADC']).agg('max').reset_index()

            df_upper_uniquecond = df_upper.groupby(['ADC']).agg('max').reset_index()

            df_max_ppm = pd.merge(left=df_lower_uniquecond, right=df_upper_uniquecond, on='ADC', how='outer', 
                                suffixes=('_%d' % lower_level,'_%d' % upper_level))

            # Loop through each row to backfill latest ppm if NaN
            for idx, row in df_max_ppm.iterrows():
                if math.isnan(row['ppm_%d' % lower_level]):
                    if idx == 0: # first row
                        df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = 1e6
                    else: # after first row
                        # If remaining rows are all NaN pad current cell with 0 else fill with previous row value
                        if df_max_ppm.loc[df_max_ppm.index > idx,'ppm_%d' % lower_level].isnull().all():
                            df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = 0
                        else:
                            df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = df_max_ppm.iloc[idx-1]['ppm_%d' % lower_level]
                    

            for idx, row in df_max_ppm.iloc[::-1].iterrows(): # Iterate backwards for the upper level
                if math.isnan(row['ppm_%d' % upper_level]):
                    if idx == len(df_max_ppm)-1: # last row
                        df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = 1e6
                    if idx < len(df_max_ppm)-1: # before last row
                        # If remaining rows are all NaN pad current cell with 0 else fill with previous row value
                        if df_max_ppm.loc[df_max_ppm.index < idx,'ppm_%d' % upper_level].isnull().all():
                            df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = 0
                        else:
                            df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = df_max_ppm.iloc[idx+1]['ppm_%d' % upper_level]

            # Find max ppm and backfill with 0 if needed
            df_max_ppm['Max_ppm'] = df_max_ppm[['ppm_%d' % lower_level, 'ppm_%d' % upper_level]].max(axis=1)
            df_max_ppm['Max_ppm'] = df_max_ppm['Max_ppm'].fillna(0)

            # Find max summed ppm and backfill with 0 if needed
            df_max_ppm['Max_sum_ppm'] = df_max_ppm[['ppm_%d' % lower_level, 'ppm_%d' % upper_level]].sum(axis=1)
            df_max_ppm['Max_sum_ppm'] = df_max_ppm['Max_sum_ppm'].fillna(0)

            lower_level_max_cond = sorted_data[lower_level][-1]
            upper_level_min_cond = sorted_data[upper_level][0]

            if lower_level_max_cond < upper_level_min_cond:
                # Xpoint ppm is 0 if max(lower level) < min(upper level) cond
                rwb.loc[key,'LEVEL_%d%d_XPOINT_ADC' % (lower_level,upper_level)] = (lower_level_max_cond + upper_level_min_cond)/2
                rwb.loc[key,'LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)] = 0

                rwb.loc[key,'LEVEL_%d%d_XPOINT_SUM_ADC' % (lower_level,upper_level)] = (lower_level_max_cond + upper_level_min_cond)/2
                rwb.loc[key,'LEVEL_%d%d_XPOINT_SUM_PPM' % (lower_level,upper_level)] = 0
            else:
                # Find the minimum ppm and its location
                min_ppm = df_max_ppm['Max_ppm'].min()
                rwb.loc[key,'LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)] = min_ppm
                min_ppm_cond = df_max_ppm.loc[df_max_ppm['Max_ppm']==min_ppm]['ADC'].mean() # Take mean if a range of ADC values
                rwb.loc[key,'LEVEL_%d%d_XPOINT_ADC' % (lower_level,upper_level)] = min_ppm_cond

                # Find the minimum sum ppm and its location
                min_sum_ppm = min(df_max_ppm['Max_sum_ppm'].min(), 1e6)
                rwb.loc[key,'LEVEL_%d%d_XPOINT_SUM_PPM' % (lower_level,upper_level)] = min_sum_ppm
                min_sum_ppm_cond = df_max_ppm.loc[df_max_ppm['Max_ppm']==min_ppm]['ADC'].mean() # Take mean if a range of ADC values
                rwb.loc[key,'LEVEL_%d%d_XPOINT_SUM_ADC' % (lower_level,upper_level)] = min_sum_ppm_cond

            # # Calculate ppm at each conduction for each level
            # for cond in interpolated_cond_range:
            #     index_upper_level = bisect.bisect_right(sorted_data[upper_level], cond)
            #     if index_upper_level == len(sorted_data[upper_level]):
            #         upper_level_ppm = 1e6
            #     else:
            #         upper_level_ppm = stats.norm.cdf(normal_quantiles[upper_level][index_upper_level]) * 1e6

            #     index_lower_level = bisect.bisect_right(sorted_data[lower_level], cond)
            #     if index_lower_level == len(sorted_data[lower_level]):
            #         lower_level_ppm = 0
            #     else:
            #         lower_level_ppm = stats.norm.cdf(-normal_quantiles[lower_level][index_lower_level]) * 1e6

            #     # rwb.loc[key,'LEVEL_%d%d_%dUS_PPM' % (lower_level,upper_level,cond)] = lower_level_ppm + upper_level_ppm
            #     # rwb.loc[key,'LEVEL_%d%d_%dADC_PPM' % (lower_level,upper_level,cond)] = cond_to_adc(lower_level_ppm, cond_range) + cond_to_adc(upper_level_ppm, cond_range)


        # Calculate sigma and min/max ADC for each level
        for level in [0,1,2,3]:
            rwb.loc[key,'LEVEL_%d_DISTSIGMA' % level] = np.std(sorted_data[level])

            rwb.loc[key,'LEVEL_%d_MAXADC' % level] = sorted_data[level][-1]
            rwb.loc[key,'LEVEL_%d_MINADC' % level] = sorted_data[level][0]

            # # Calculate ppm at each ADC for each level
            # for cond in interpolated_cond_range:
            #     index = bisect.bisect_right(sorted_data[level], cond)
            #     if index == len(sorted_data[level]):
            #         ppm = 1e6
            #     else:
            #         ppm = stats.norm.cdf(normal_quantiles[level][index]) * 1e6
            #     rwb.loc[key,'LEVEL_%d_%dUS_PPM' % (level,cond)] = ppm

            # Calculate ppm at each ADC code for each level
            for adc in range(64):
                index = bisect.bisect_right(sorted_data[level], adc)
                if index == len(sorted_data[level]):
                    ppm = 1e6
                else:
                    ppm = stats.norm.cdf(normal_quantiles[level][index]) * 1e6
                rwb.loc[key,'LEVEL_%d_%dADC_PPM' % (level,adc)] = ppm

    
    # Make sure df columns are flat
    rwb.columns = ['_'.join(col) for col in rwb.columns.to_flat_index()]

    return rwb


# Function to generate subplots of NQ distributions, data_list is a dictionary with the first level the readout name and secondary level the bit pattern (aka level).
# Titles is the keys of data in the order they wish to be plotted.
# Returns a dictionary of read window budget (rwb) for each key in the data plotted and the interpolated ppm distributions
# cpm_range is a tuple of (adc_code_0_cond, adc_code_63_cond)
def rwb_intcond_overlaylevelnqplot_OLD(data_list, titles = ['plot 1','plot 2','plot 3','plot 4'], ref_ppm_list = [170,500], xpoint_levels_list = [(0,1),(1,2),(2,3)], 
                                   enable_plots = False, plot_ref_lines = True, cond_range=(30,160)):
    if enable_plots:
        if len(titles) > 4:
            num_rows = math.ceil(len(titles)/2)
            fig, axs = plt.subplots(num_rows, 2, figsize=(14, 2+num_rows*5.7))  # 2 row, 2 columns of subplots
            axs = axs.flatten()
        elif len(titles) > 2:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 row, 2 columns of subplots
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 row, 2 columns of subplots
            axs = axs.flatten()

        # Colors for each line in the plots
        colors = ['r', 'b', 'g', 'm']

    def cond_to_adc(cond, cond_range):
        # cond = input * ((cond_63 - cond_0)/63) + cond_0
        # input = (cond - cond_0) * 63 / (cond_63 - cond_0)
        
        adc_code = round((cond - cond_range[0]) * 63 / (cond_range[1] - cond_range[0]))
        
        if adc_code < 0:
            print('Warning, determined conductance < 0, forcing adc code to 0')
            adc_code = 0
        elif adc_code > 63:
            print('Warning, determined conductance < 0, forcing adc code to 0')
            adc_code = 63
        
        return adc_code

    # Reference ppm list is 0, -1, -2, -2.5, -3 sigma and whatever inputs passed for plot referencing
    sigma_list = [0, -1, -2, -2.5, -3]
    ref_ppmsigma_list = [('sigma', x) for x in sigma_list]
    ref_ppmsigma_list = ref_ppmsigma_list + [('ppm', x) for x in ref_ppm_list]

    # Convert sigma list into a string for column names
    str_sigma_list = [str(x).replace('.','P') for x in sigma_list]
    str_sigma_list = [x.replace('-','M') for x in str_sigma_list]

    # Initialize rwb dataframe
    min_cond = 100 # in uS
    max_cond = 0 # in uS
    for key in data_list.keys():
        for level in data_list[key].keys():
            min_cond_temp = np.min(data_list[key][level])
            max_cond_temp = np.max(data_list[key][level])
            if min_cond_temp < min_cond:
                min_cond = min_cond_temp
            if max_cond_temp > max_cond:
                max_cond = max_cond_temp
    interpolated_cond_range = np.arange(int(min_cond/2)*2, int(max_cond/2)*2+2, 2)
    rwb = pd.DataFrame(np.nan, index=titles, columns=[['LEVEL_%d%d_XPOINT_PPM' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        ['LEVEL_%d%d_XPOINT_COND' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        ['LEVEL_%d%d_XPOINT_ADC' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        list(chain.from_iterable([['LEVEL_%d%d_%sSIGMA_CONDRWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([['LEVEL_%d%d_%dPPM_CONDRWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in ref_ppm_list])) + 
                        list(chain.from_iterable([['LEVEL_%d%d_%sSIGMA_ADCRWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([['LEVEL_%d%d_%dPPM_ADCRWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in ref_ppm_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%sSIGMA_COND' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%sSIGMA_COND' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%dPPM_COND' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%dPPM_COND' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in ref_ppm_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%dPPM_ADC' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%dPPM_ADC' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in ref_ppm_list])) + 
                        list(chain.from_iterable([['LEVEL_%d_MAXCOND' % level, 'LEVEL_%d_MINCOND' % level] for level in [0,1,2,3]])) + 
                        ['LEVEL_%d_DISTSIGMA' % level for level in [0,1,2,3]] + 
                        ['LEVEL_0_%dUS_PPM' % cond for cond in interpolated_cond_range] + ['LEVEL_1_%dUS_PPM' % cond for cond in interpolated_cond_range] + ['LEVEL_2_%dUS_PPM' % cond for cond in interpolated_cond_range] + ['LEVEL_3_%dUS_PPM' % cond for cond in interpolated_cond_range] + 
                        # ['LEVEL_01_%dUS_PPM' % cond for cond in interpolated_cond_range] + ['LEVEL_12_%dUS_PPM' % cond for cond in interpolated_cond_range] + ['LEVEL_23_%dUS_PPM' % cond for cond in interpolated_cond_range] +
                        ['LEVEL_0_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_1_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_2_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_3_%dADC_PPM' % cond for cond in range(64)], 
                        # ['LEVEL_01_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_12_%dADC_PPM' % cond for cond in range(64)] + ['LEVEL_23_%dADC_PPM' % cond for cond in range(64)],
                        ])
    
    for i, key in enumerate(titles):
        # First subplot for `rowbar_1d`
        j = 0
        sorted_data = [np.nan, np.nan, np.nan, np.nan]
        normal_quantiles = [np.nan, np.nan, np.nan, np.nan]
        for level, plot_data in data_list[key].items():
            sorted_data[j] = np.sort(plot_data)
            quantiles = np.linspace(0, 1, len(sorted_data[j]))
            normal_quantiles[j] = stats.norm.ppf(quantiles)
            if enable_plots:
                axs[i].plot(sorted_data[j], normal_quantiles[j], marker='.', linestyle='-', color=colors[j], label='LEVEL %d' % level)
            j += 1

        if enable_plots:
            # Draw reference lines
            x_min, x_max = axs[i].get_xlim()
            x_pos = (x_min + x_max)/2 # draw at mid point
            if plot_ref_lines:
                for ppm in ref_ppm_list:
                    ref_sigma = stats.norm.ppf(ppm/1e6)
                    
                    axs[i].axhline(y=ref_sigma, color='gray', linestyle='--')
                    axs[i].text(x=x_pos, y=ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label
                    
                    axs[i].axhline(y=-ref_sigma, color='gray', linestyle='--')
                    axs[i].text(x=x_pos, y=-ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label

            # Customize first subplot
            axs[i].set_xlabel('Conductance [uS]')
            axs[i].set_ylabel('Normal Quantiles')
            axs[i].set_title('Overlay of Distributions for %s' % key)
            axs[i].legend() # [ 'Level %d' % x for x in range(4)]
            axs[i].grid(True)

        # if output_nq_data:
        #     # Convert 
        #     nq_output = 

        ### Calculate RWB ###
        # xpoint_levels_list = [(0,1),(1,2),(2,3),(0,3)] # List of intersection levels at which to calculate rwb
        for lower_level, upper_level in xpoint_levels_list:
            for stat_type, stat_value in ref_ppmsigma_list:
                # Convert ppm into sigma
                if stat_type == 'ppm':
                    ref_sigma = stats.norm.ppf(stat_value/1e6)
                elif stat_type =='sigma':
                    ref_sigma = stat_value

                # Find conductance at ppm for lower level, sigma is positive
                index_lower_level = bisect.bisect_right(normal_quantiles[lower_level], abs(ref_sigma))
                cond_lower_level = sorted_data[lower_level][index_lower_level]

                # Find conductance at ppm for upper level, sigma is negative
                index_upper_level = bisect.bisect_right(normal_quantiles[upper_level], -abs(ref_sigma))
                cond_upper_level = sorted_data[upper_level][index_upper_level]

                # Save RWB and lower and upper conductances for this particular test and xpoint
                if stat_type == 'ppm':
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_COND' % (lower_level, upper_level, lower_level, round(stat_value))] = cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_COND' % (lower_level, upper_level, upper_level, round(stat_value))] = cond_upper_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_ADC' % (lower_level, upper_level, lower_level, round(stat_value))] = cond_to_adc(cond_lower_level, cond_range)
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_ADC' % (lower_level, upper_level, upper_level, round(stat_value))] = cond_to_adc(cond_upper_level, cond_range)
                    rwb.loc[key,'LEVEL_%d%d_%dPPM_CONDRWB' % (lower_level, upper_level, round(stat_value))] = cond_upper_level - cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%dPPM_ADCRWB' % (lower_level, upper_level, round(stat_value))] = cond_to_adc(cond_upper_level, cond_range) - cond_to_adc(cond_lower_level, cond_range)
                elif stat_type =='sigma':
                    print_sigma = str(ref_sigma).replace('.','P') # replace period with 'P'
                    print_sigma = print_sigma.replace('-','M') # replace minus sign with 'M'
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_COND' % (lower_level, upper_level, lower_level, print_sigma)] = cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_COND' % (lower_level, upper_level, upper_level, print_sigma)] = cond_upper_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower_level, upper_level, lower_level, print_sigma)] = cond_to_adc(cond_lower_level, cond_range)
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_ADC' % (lower_level, upper_level, upper_level, print_sigma)] = cond_to_adc(cond_upper_level, cond_range)
                    rwb.loc[key,'LEVEL_%d%d_%sSIGMA_CONDRWB' % (lower_level, upper_level, print_sigma)] = cond_upper_level - cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%sSIGMA_ADCRWB' % (lower_level, upper_level, print_sigma)] = cond_to_adc(cond_upper_level, cond_range) - cond_to_adc(cond_lower_level, cond_range)

            # Calculate xpoint conductance and ppm
            # Sum ppm from lower and upper level at matched conductance
            df_lower = pd.DataFrame({'Conductance': sorted_data[lower_level], 
                                            'ppm': stats.norm.cdf(-normal_quantiles[lower_level])*1e6})
            df_upper = pd.DataFrame({'Conductance': sorted_data[upper_level], 
                                            'ppm': stats.norm.cdf(normal_quantiles[upper_level])*1e6})
            
            # Find highest ppm for each conductance value then combine into a dataframe
            df_lower_uniquecond = df_lower.groupby(['Conductance']).agg('max').reset_index()

            df_upper_uniquecond = df_upper.groupby(['Conductance']).agg('max').reset_index()

            df_max_ppm = pd.merge(left=df_lower_uniquecond, right=df_upper_uniquecond, on='Conductance', how='outer', 
                                suffixes=('_%d' % lower_level,'_%d' % upper_level))

            # Loop through each row to backfill latest ppm if NaN
            for idx, row in df_max_ppm.iterrows():
                if math.isnan(row['ppm_%d' % lower_level]):
                    if idx == 0: # first row
                        df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = 1e6
                    else: # after first row
                        # If remaining rows are all NaN pad current cell with 0 else fill with previous row value
                        if df_max_ppm.loc[df_max_ppm.index > idx,'ppm_%d' % lower_level].isnull().all():
                            df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = 0
                        else:
                            df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = df_max_ppm.iloc[idx-1]['ppm_%d' % lower_level]
                    

            for idx, row in df_max_ppm.iloc[::-1].iterrows(): # Iterate backwards for the upper level
                if math.isnan(row['ppm_%d' % upper_level]):
                    if idx == len(df_max_ppm)-1: # last row
                        df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = 1e6
                    if idx < len(df_max_ppm)-1: # before last row
                        # If remaining rows are all NaN pad current cell with 0 else fill with previous row value
                        if df_max_ppm.loc[df_max_ppm.index < idx,'ppm_%d' % upper_level].isnull().all():
                            df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = 0
                        else:
                            df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = df_max_ppm.iloc[idx+1]['ppm_%d' % upper_level]

            # Find max ppm and backfill with 0 if needed
            df_max_ppm['Max_ppm'] = df_max_ppm[['ppm_%d' % lower_level, 'ppm_%d' % upper_level]].max(axis=1)
            df_max_ppm['Max_ppm'] = df_max_ppm['Max_ppm'].fillna(0)

            lower_level_max_cond = sorted_data[lower_level][-1]
            upper_level_min_cond = sorted_data[upper_level][0]

            if lower_level_max_cond < upper_level_min_cond:
                # Xpoint ppm is 0 if max(lower level) < min(upper level) cond
                rwb.loc[key,'LEVEL_%d%d_XPOINT_COND' % (lower_level,upper_level)] = (lower_level_max_cond + upper_level_min_cond)/2
                rwb.loc[key,'LEVEL_%d%d_XPOINT_ADC' % (lower_level,upper_level)] = (cond_to_adc(lower_level_max_cond, cond_range) + cond_to_adc(upper_level_min_cond, cond_range))/2
                rwb.loc[key,'LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)] = 0
            else:
                # Find the minimum ppm and its location
                min_ppm = df_max_ppm['Max_ppm'].min()
                rwb.loc[key,'LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)] = min_ppm
                min_ppm_cond = df_max_ppm.loc[df_max_ppm['Max_ppm']==min_ppm]['Conductance'].mean() # Take mean if a range of conductance values
                rwb.loc[key,'LEVEL_%d%d_XPOINT_COND' % (lower_level,upper_level)] = min_ppm_cond
                rwb.loc[key,'LEVEL_%d%d_XPOINT_ADC' % (lower_level,upper_level)] = cond_to_adc(min_ppm_cond, cond_range)

            # Calculate ppm at each conduction for each level
            for cond in interpolated_cond_range:
                index_upper_level = bisect.bisect_right(sorted_data[upper_level], cond)
                if index_upper_level == len(sorted_data[upper_level]):
                    upper_level_ppm = 1e6
                else:
                    upper_level_ppm = stats.norm.cdf(normal_quantiles[upper_level][index_upper_level]) * 1e6

                index_lower_level = bisect.bisect_right(sorted_data[lower_level], cond)
                if index_lower_level == len(sorted_data[lower_level]):
                    lower_level_ppm = 0
                else:
                    lower_level_ppm = stats.norm.cdf(-normal_quantiles[lower_level][index_lower_level]) * 1e6

                # rwb.loc[key,'LEVEL_%d%d_%dUS_PPM' % (lower_level,upper_level,cond)] = lower_level_ppm + upper_level_ppm
                # rwb.loc[key,'LEVEL_%d%d_%dADC_PPM' % (lower_level,upper_level,cond)] = cond_to_adc(lower_level_ppm, cond_range) + cond_to_adc(upper_level_ppm, cond_range)


        # Calculate sigma and min/max conductance for each level
        for level in [0,1,2,3]:
            rwb.loc[key,'LEVEL_%d_DISTSIGMA' % level] = np.std(sorted_data[level])

            rwb.loc[key,'LEVEL_%d_MAXCOND' % level] = sorted_data[level][-1]
            rwb.loc[key,'LEVEL_%d_MINCOND' % level] = sorted_data[level][0]

            # Calculate ppm at each conductance for each level
            for cond in interpolated_cond_range:
                index = bisect.bisect_right(sorted_data[level], cond)
                if index == len(sorted_data[level]):
                    ppm = 1e6
                else:
                    ppm = stats.norm.cdf(normal_quantiles[level][index]) * 1e6
                rwb.loc[key,'LEVEL_%d_%dUS_PPM' % (level,cond)] = ppm

            # Calculate ppm at each ADC code for each level
            for adc in range(64):
                cond = adc * (cond_range[1] - cond_range[0]) / 63 + cond_range[0]
                index = bisect.bisect_right(sorted_data[level], cond)
                if index == len(sorted_data[level]):
                    ppm = 1e6
                else:
                    ppm = stats.norm.cdf(normal_quantiles[level][index]) * 1e6
                rwb.loc[key,'LEVEL_%d_%dADC_PPM' % (level,adc)] = ppm

    
    # Make sure df columns are flat
    rwb.columns = ['_'.join(col) for col in rwb.columns.to_flat_index()]

    return rwb

def nq_plotsandrwb_overlaylevel_OLD(data_list, titles = ['plot 1','plot 2','plot 3','plot 4'], plot_ref_ppm_list = [170,500], xpoint_levels_list = [(0,1),(1,2),(2,3)], enable_plots = False):
    if enable_plots:
        if len(titles) > 4:
            num_rows = math.ceil(len(titles)/2)
            fig, axs = plt.subplots(num_rows, 2, figsize=(14, 2+num_rows*5.7))  # 2 row, 2 columns of subplots
            axs = axs.flatten()
        elif len(titles) > 2:
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2 row, 2 columns of subplots
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 2 row, 2 columns of subplots

        # Colors for each line in the plots
        colors = ['r', 'b', 'g', 'm']

    # Reference ppm list is 0, -1, -2, -2.5, -3 sigma and whatever inputs passed for plot referencing
    sigma_list = [0, -1, -2, -2.5, -3]
    ref_ppmsigma_list = [('sigma', x) for x in sigma_list]
    ref_ppmsigma_list = ref_ppmsigma_list + [('ppm', x) for x in plot_ref_ppm_list]

    # Convert sigma list into a string for column names
    str_sigma_list = [str(x).replace('.','P') for x in sigma_list]
    str_sigma_list = [x.replace('-','M') for x in str_sigma_list]

    # Initialize dataframe
    rwb = pd.DataFrame(np.nan, index=titles, columns=[['LEVEL_%d%d_XPOINT_PPM' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        ['LEVEL_%d%d_XPOINT_COND' % (lower,upper) for (lower,upper) in xpoint_levels_list] + 
                        list(chain.from_iterable([['LEVEL_%d%d_%sSIGMA_RWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([['LEVEL_%d%d_%dPPM_RWB' % (lower,upper,ppm) for (lower,upper) in xpoint_levels_list] for ppm in plot_ref_ppm_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%sSIGMA_COND' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%sSIGMA_COND' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in str_sigma_list])) + 
                        list(chain.from_iterable([list(chain.from_iterable([['LEVEL_%d%d_%d_%dPPM_COND' % (lower,upper,lower,ppm), 'LEVEL_%d%d_%d_%dPPM_COND' % (lower,upper,upper,ppm)] for (lower,upper) in xpoint_levels_list])) for ppm in plot_ref_ppm_list])) + 
                        list(chain.from_iterable([['LEVEL_%d_MAXCOND' % level, 'LEVEL_%d_MINCOND' % level] for level in [0,1,2,3]])) + 
                        ['LEVEL_%d_DISTSIGMA' % level for level in [0,1,2,3]]
                        ])
    
    for i, key in enumerate(titles):
        # First subplot for `rowbar_1d`
        j = 0
        sorted_data = [np.nan, np.nan, np.nan, np.nan]
        normal_quantiles = [np.nan, np.nan, np.nan, np.nan]
        for level, plot_data in data_list[key].items():
            sorted_data[j] = np.sort(plot_data)
            quantiles = np.linspace(0, 1, len(sorted_data[j]))
            normal_quantiles[j] = stats.norm.ppf(quantiles)
            if enable_plots:
                axs[i].plot(sorted_data[j], normal_quantiles[j], marker='.', linestyle='-', color=colors[j], label='LEVEL %d' % level)
            j += 1

        if enable_plots:
            # Draw reference lines
            x_min, x_max = axs[i].get_xlim()
            x_pos = (x_min + x_max)/2 # draw at mid point
            for ppm in plot_ref_ppm_list:
                ref_sigma = stats.norm.ppf(ppm/1e6)
                
                axs[i].axhline(y=ref_sigma, color='gray', linestyle='--')
                axs[i].text(x=x_pos, y=ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label
                
                axs[i].axhline(y=-ref_sigma, color='gray', linestyle='--')
                axs[i].text(x=x_pos, y=-ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label

            # Customize first subplot
            axs[i].set_xlabel('Conductance [uS]')
            axs[i].set_ylabel('Normal Quantiles')
            axs[i].set_title('Overlay of Distributions for %s' % key)
            axs[i].legend() # [ 'Level %d' % x for x in range(4)]
            axs[i].grid(True)

        ### Calculate RWB ###
        # xpoint_levels_list = [(0,1),(1,2),(2,3),(0,3)] # List of intersection levels at which to calculate rwb
        for lower_level, upper_level in xpoint_levels_list:
            for stat_type, stat_value in ref_ppmsigma_list:
                # Convert ppm into sigma
                if stat_type == 'ppm':
                    ref_sigma = stats.norm.ppf(stat_value/1e6)
                elif stat_type =='sigma':
                    ref_sigma = stat_value

                # Find conductance at ppm for lower level, sigma is positive
                index_lower_level = bisect.bisect_right(normal_quantiles[lower_level], abs(ref_sigma))
                cond_lower_level = sorted_data[lower_level][index_lower_level]

                # Find conductance at ppm for upper level, sigma is negative
                index_upper_level = bisect.bisect_right(normal_quantiles[upper_level], -abs(ref_sigma))
                cond_upper_level = sorted_data[upper_level][index_upper_level]

                # Save RWB and lower and upper conductances for this particular test and xpoint
                if stat_type == 'ppm':
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_COND' % (lower_level, upper_level, lower_level, round(stat_value))] = cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%dPPM_COND' % (lower_level, upper_level, upper_level, round(stat_value))] = cond_upper_level
                    rwb.loc[key,'LEVEL_%d%d_%dPPM_RWB' % (lower_level, upper_level, round(stat_value))] = cond_upper_level - cond_lower_level
                elif stat_type =='sigma':
                    print_sigma = str(ref_sigma).replace('.','P') # replace period with 'P'
                    print_sigma = print_sigma.replace('-','M') # replace minus sign with 'M'
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_COND' % (lower_level, upper_level, lower_level, print_sigma)] = cond_lower_level
                    rwb.loc[key,'LEVEL_%d%d_%d_%sSIGMA_COND' % (lower_level, upper_level, upper_level, print_sigma)] = cond_upper_level
                    rwb.loc[key,'LEVEL_%d%d_%sSIGMA_RWB' % (lower_level, upper_level, print_sigma)] = cond_upper_level - cond_lower_level

            # Calculate xpoint conductance and ppm
            # Sum ppm from lower and upper level at matched conductance
            df_lower = pd.DataFrame({'Conductance': sorted_data[lower_level], 
                                            'ppm': stats.norm.cdf(-normal_quantiles[lower_level])*1e6})
            df_upper = pd.DataFrame({'Conductance': sorted_data[upper_level], 
                                            'ppm': stats.norm.cdf(normal_quantiles[upper_level])*1e6})
            
            # Find highest ppm for each conductance value then combine into a dataframe
            df_lower_uniquecond = df_lower.groupby(['Conductance']).agg('max').reset_index()

            df_upper_uniquecond = df_upper.groupby(['Conductance']).agg('max').reset_index()

            df_max_ppm = pd.merge(left=df_lower_uniquecond, right=df_upper_uniquecond, on='Conductance', how='outer', 
                                suffixes=('_%d' % lower_level,'_%d' % upper_level))

            # Loop through each row to backfill latest ppm if NaN
            for idx, row in df_max_ppm.iterrows():
                if math.isnan(row['ppm_%d' % lower_level]):
                    if idx == 0: # first row
                        df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = 1e6
                    else: # after first row
                        # If remaining rows are all NaN pad current cell with 0 else fill with previous row value
                        if df_max_ppm.loc[df_max_ppm.index > idx,'ppm_%d' % lower_level].isnull().all():
                            df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = 0
                        else:
                            df_max_ppm.loc[idx, 'ppm_%d' % lower_level] = df_max_ppm.iloc[idx-1]['ppm_%d' % lower_level]
                    

            for idx, row in df_max_ppm.iloc[::-1].iterrows(): # Iterate backwards for the upper level
                if math.isnan(row['ppm_%d' % upper_level]):
                    if idx == len(df_max_ppm)-1: # last row
                        df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = 1e6
                    if idx < len(df_max_ppm)-1: # before last row
                        # If remaining rows are all NaN pad current cell with 0 else fill with previous row value
                        if df_max_ppm.loc[df_max_ppm.index < idx,'ppm_%d' % upper_level].isnull().all():
                            df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = 0
                        else:
                            df_max_ppm.loc[idx, 'ppm_%d' % upper_level] = df_max_ppm.iloc[idx+1]['ppm_%d' % upper_level]

            # Find max ppm and backfill with 0 if needed
            df_max_ppm['Max_ppm'] = df_max_ppm[['ppm_%d' % lower_level, 'ppm_%d' % upper_level]].max(axis=1)
            df_max_ppm['Max_ppm'] = df_max_ppm['Max_ppm'].fillna(0)

            # Find the minimum ppm and its location
            min_ppm = df_max_ppm['Max_ppm'].min()
            rwb.loc[key,'LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)] = min_ppm
            min_ppm_cond = df_max_ppm.loc[df_max_ppm['Max_ppm']==min_ppm]['Conductance'].mean() # Take mean if a range of conductance values
            rwb.loc[key,'LEVEL_%d%d_XPOINT_COND' % (lower_level,upper_level)] = min_ppm_cond


        # Calculate sigma and min/max conductance for each level
        for level in [0,1,2,3]:
            rwb.loc[key,'LEVEL_%d_DISTSIGMA' % level] = np.std(sorted_data[level])

            rwb.loc[key,'LEVEL_%d_MAXCOND' % level] = sorted_data[level][-1]
            rwb.loc[key,'LEVEL_%d_MINCOND' % level] = sorted_data[level][0]

    
    # Make sure df columns are flat
    rwb.columns = ['_'.join(col) for col in rwb.columns.to_flat_index()]

    return rwb

def plot_rwb(rwb, ref_ppm_list = [170,500], logscale_ppm = False, xpoint_levels_list = [(0,1),(1,2),(2,3)], overlay_var = None):
    # Plot RWB info: Subplots for xpoint ppm, xpoint cond, 170ppm rwb and 1000ppm rwb at each window
    # Set up a figure and two subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    [axs_xpoint_ppm, axs_xpoint_cond, axs_ppm1_rwb, axs_ppm2_rwb] = axs.flatten()

    if overlay_var is None:
        for lower_level, upper_level in xpoint_levels_list:
            # Plot xpoint ppm
            axs_xpoint_ppm.plot(rwb.index, rwb['LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)].values, marker='o', label='Window %d%d' % (lower_level,upper_level))
            for ppm in ref_ppm_list:
                axs_xpoint_ppm.axhline(y=ppm, color='gray', linestyle='--')
            
            if logscale_ppm:
                axs_xpoint_ppm.set_yscale('log')
                _, upper_ylim = axs_xpoint_ppm.get_ylim()
                axs_xpoint_ppm.set_ylim([0.8, 100000])
            
            axs_xpoint_ppm.set_ylabel('Xpoint PPM')
            axs_xpoint_ppm.legend()
            axs_xpoint_ppm.grid(True)

            # Plot xpoint cond
            axs_xpoint_cond.plot(rwb.index, rwb['LEVEL_%d%d_XPOINT_ADC' % (lower_level,upper_level)].values, marker='o', label='Window %d%d' % (lower_level,upper_level))
            axs_xpoint_cond.set_ylabel('Xpoint Conductance [uS]')
            axs_xpoint_cond.legend()
            axs_xpoint_cond.grid(True)

            # Plot 170ppm rwb
            axs_ppm1_rwb.plot(rwb.index, rwb['LEVEL_%d%d_%dPPM_ADCRWB' % (lower_level,upper_level,ref_ppm_list[0])].values, marker='o', label='Window %d%d' % (lower_level,upper_level))
            axs_ppm1_rwb.axhline(y=0, color='gray', linestyle='--')
            axs_ppm1_rwb.set_ylabel(f'{ref_ppm_list[0]}ppm RWB [uS]')
            axs_ppm1_rwb.legend()
            axs_ppm1_rwb.grid(True)

            # Plot 1000ppm rwb
            axs_ppm2_rwb.plot(rwb.index, rwb['LEVEL_%d%d_%dPPM_ADCRWB' % (lower_level,upper_level,ref_ppm_list[1])].values, marker='o', label='Window %d%d' % (lower_level,upper_level))
            axs_ppm2_rwb.axhline(y=0, color='gray', linestyle='--')
            axs_ppm2_rwb.set_ylabel(f'{ref_ppm_list[1]}ppm RWB [uS]')
            axs_ppm2_rwb.legend()
            axs_ppm2_rwb.grid(True)

        # Show the plot
        if type(rwb.index[0])=='str':
            if len(rwb.index)*len(rwb.index[0])>100:
                axs_xpoint_ppm.set_xticklabels(rwb.index, rotation=70)
                axs_xpoint_cond.set_xticklabels(rwb.index, rotation=70)
                axs_ppm1_rwb.set_xticklabels(rwb.index, rotation=70)
                axs_ppm2_rwb.set_xticklabels(rwb.index, rotation=70)
        plt.tight_layout()
        plt.show()
    else:
        print('Change the first entry in the xpoint_levels_list to the desired window to plot')
        lower_level, upper_level = xpoint_levels_list[0]
        for var in rwb[overlay_var].unique():
            # Plot xpoint ppm
            axs_xpoint_ppm.plot(rwb.loc[rwb[overlay_var]==var].index, rwb.loc[rwb[overlay_var]==var]['LEVEL_%d%d_XPOINT_PPM' % (lower_level,upper_level)].values, marker='o', label=f'Window {lower_level}{upper_level} with {overlay_var}={var}')
            for ppm in ref_ppm_list:
                axs_xpoint_ppm.axhline(y=ppm, color='gray', linestyle='--')
            
            if logscale_ppm:
                axs_xpoint_ppm.set_yscale('log')
                _, upper_ylim = axs_xpoint_ppm.get_ylim()
                axs_xpoint_ppm.set_ylim([0.8, 100000])
            
            axs_xpoint_ppm.set_ylabel('Xpoint PPM')
            axs_xpoint_ppm.legend()
            axs_xpoint_ppm.grid(True)

            # Plot xpoint cond
            axs_xpoint_cond.plot(rwb.loc[rwb[overlay_var]==var].index, rwb.loc[rwb[overlay_var]==var]['LEVEL_%d%d_XPOINT_COND' % (lower_level,upper_level)].values, marker='o', label=f'Window {lower_level}{upper_level} with {overlay_var}={var}')
            axs_xpoint_cond.set_ylabel('Xpoint Conductance [uS]')
            axs_xpoint_cond.legend()
            axs_xpoint_cond.grid(True)

            # Plot 170ppm rwb
            axs_ppm1_rwb.plot(rwb.loc[rwb[overlay_var]==var].index, rwb.loc[rwb[overlay_var]==var]['LEVEL_%d%d_%dPPM_RWB' % (lower_level,upper_level,ref_ppm_list[0])].values, marker='o', label=f'Window {lower_level}{upper_level} with {overlay_var}={var}')
            axs_ppm1_rwb.axhline(y=0, color='gray', linestyle='--')
            axs_ppm1_rwb.set_ylabel(f'{ref_ppm_list[0]}ppm RWB [uS]')
            axs_ppm1_rwb.legend()
            axs_ppm1_rwb.grid(True)

            # Plot 1000ppm rwb
            axs_ppm2_rwb.plot(rwb.loc[rwb[overlay_var]==var].index, rwb.loc[rwb[overlay_var]==var]['LEVEL_%d%d_%dPPM_RWB' % (lower_level,upper_level,ref_ppm_list[1])].values, marker='o', label=f'Window {lower_level}{upper_level} with {overlay_var}={var}')
            axs_ppm2_rwb.axhline(y=0, color='gray', linestyle='--')
            axs_ppm2_rwb.set_ylabel(f'{ref_ppm_list[1]}ppm RWB [uS]')
            axs_ppm2_rwb.legend()
            axs_ppm2_rwb.grid(True)

        # Show the plot
        if type(rwb.index[0])=='str':
            if len(rwb.index)*len(rwb.index[0])>100:
                axs_xpoint_ppm.set_xticklabels(rwb.index, rotation=70)
                axs_xpoint_cond.set_xticklabels(rwb.index, rotation=70)
                axs_ppm1_rwb.set_xticklabels(rwb.index, rotation=70)
                axs_ppm2_rwb.set_xticklabels(rwb.index, rotation=70)
        plt.tight_layout()
        plt.show()

# This function overlaps various readout conditions by level and plots their NQ distributions
def nq_plots_overlayreadout(data_list, titles = ['plot 1','plot 2','plot 3','plot 4'], ref_ppm_list = [170,1000]):
    # Set up the figure and subplots
    fig, axs = plt.subplots(1, 4, figsize=(14, 6))  # 1 row, 2 columns of subplots

    # Colors for each line in the plots
    colors = list(mcolors.BASE_COLORS.keys())
    # Don't import white
    index_of_w = colors.index('w')
    color_w = colors.pop(index_of_w)
    # Place red as the first color
    index_of_r = colors.index('r')
    color_r = colors.pop(index_of_r)
    colors.insert(0, color_r)

    # Truncate list of keys to plot if longer than color list
    if len(titles)>len(colors):
        print('Warning: too many keys to plot only keeping first %d keys' % len(colors))
        titles = titles[:len(colors)]

    for level in range(4):
        # First subplot for level 0
        j = 0
        for key in titles:
            sorted_data = np.sort(data_list[key][level])
            quantiles = np.linspace(0, 1, len(sorted_data))
            normal_quantiles = stats.norm.ppf(quantiles)
            axs[level].plot(sorted_data, normal_quantiles, marker='.', linestyle='-', color=colors[j], label=key)
            j += 1

        # Draw reference lines
        x_min, x_max = axs[level].get_xlim()
        x_pos = (x_min + x_max)/2 # draw at mid point
        for ppm in ref_ppm_list:
            ref_sigma = stats.norm.ppf(ppm/1e6)
            
            axs[level].axhline(y=ref_sigma, color='gray', linestyle='--')
            axs[level].text(x=x_pos, y=ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label
            
            axs[level].axhline(y=-ref_sigma, color='gray', linestyle='--')
            axs[level].text(x=x_pos, y=-ref_sigma+0.15, s='%d ppm' % ppm, va='center', ha='left', color='gray')  # Add custom label

        # Customize subplot
        axs[level].set_xlabel('Conductance [uS]')
        axs[level].set_ylabel('Normal Quantiles')
        axs[level].set_title('Overlay of Distributions for Level %d' % level)
        if level ==3:
            axs[level].legend()
        axs[level].grid(True)


    plt.tight_layout()
    plt.show()

# Plot cond vs. bit sigma overlaying level 0/1/2/3 grouped by custom column
def rwb_overlay_levels_nqplot(rwb, ref_ppm=170, groupby_col='IO', single_col_subplot=True):
    if len(rwb) != rwb[groupby_col].nunique():
        raise Exception('Each row must be a unique value in the groupby column')

    col_names = [x for x in rwb.columns if re.search('^LEVEL_[0-3]_.*ADC_PPM$',x)]
    window_ppm_data = rwb[col_names]

    if single_col_subplot:
        fig, axs = plt.subplots(len(rwb), 1, figsize=(5, 3*len(rwb)))
    else:
        n_rows_to_plot = math.ceil(len(rwb)/2)
        fig, axs = plt.subplots(n_rows_to_plot, 2, figsize=(10, 3*n_rows_to_plot))  # 2 row, 2 columns of subplots
    axs = axs.flatten()

    # Find range of conductance
    cond_list = []
    for col in col_names:
        cond_list.append(int(col.split('_')[2].split('ADC')[0]))
    cond_list = list(set(cond_list))
    cond_list.sort()
    # cond_min = min(cond_list)
    # cond_max = max(cond_list)

    for i in range(len(rwb)):
        # Extract column names (x-axis) and values (y-axis)
        # x = window_ppm_data[[x for x in window_ppm_data if '01' in x]].columns
        x = {}
        y = {}
        # Initiliaze cond rnage for each level
        for j in range(4):
            x[j] = cond_list

        x[0] = [0] + x[0]
        y[0] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_0' in x]].iloc[i]
        y[0] = [stats.norm.ppf(1/(64*1296))] + [stats.norm.ppf(x/1e6) for x in y[0]]

        y[1] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_1' in x]].iloc[i]
        y[1] = [stats.norm.ppf(x/1e6) for x in y[1]]

        y[2] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_2' in x]].iloc[i]
        y[2] = [stats.norm.ppf(x/1e6) for x in y[2]]

        # x[3] = cond_list
        # if window_ppm_data['LEVEL_3_63ADC'].iloc[i]==1e6:
        #     window_ppm_data.iloc[i,'LEVEL_3_63ADC'] = (64*1296-1)/(64*1296)
        y[3] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_3' in x]].iloc[i]
        y[3] = [stats.norm.ppf(x/1e6) for x in y[3]]

        # If all y_3 is np.inf, make last point lowest sigma then add point for highest sigma
        for level in range(4):
            # x_temp = x[level]
            # y_temp = y[level]
            if np.all(np.abs(y[level])==np.inf):
                x[level] = [63,63]
                y[level] = [stats.norm.ppf((1)/(64*1296)),stats.norm.ppf((64*1296-1)/(64*1296))]

            # If there are np.inf values, pad the ones before the data with lowest sigma and the ones after the data with highest sigma
            elif np.any(np.abs(y[level])==np.inf):
                reached_data = False
                for j in range(len(y[level])):
                    if abs(y[level][j])!=np.inf:
                        reached_data = True

                    if (reached_data==False) and (abs(y[level][j])==np.inf):
                        y[level][j] = stats.norm.ppf((1)/(64*1296))
                    elif reached_data and (abs(y[level][j])==np.inf):
                        y[level][j] = stats.norm.ppf((64*1296-1)/(64*1296))

            # Plot
            axs[i].plot(x[level], y[level], marker='o', linestyle='-', label = f'Level {level}')
                

        # Plot settings
        # axs[i].plot(x_0, y_0, marker='o', linestyle='-', label = 'Level 0')
        # axs[i].plot(x, y_1, marker='o', linestyle='-', label = 'Level 1')
        # axs[i].plot(x, y_2, marker='o', linestyle='-', label = 'Level 2')
        # axs[i].plot(x_3, y_3, marker='o', linestyle='-', label = 'Level 3')
        axs[i].set_title(f'{groupby_col}={rwb[groupby_col].values[i]}')
        axs[i].set_xlabel("ADC Code")
        axs[i].set_ylabel("Sigma")
        axs[i].set_xlim([-1,64])
        axs[i].set_ylim([-4,4])
        axs[i].axhline(stats.norm.ppf(ref_ppm/1e6), color='gray', linestyle='--')
        axs[i].axhline(-stats.norm.ppf(ref_ppm/1e6), color='gray', linestyle='--')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

# Plot cond vs. bit sigma for level 0/1/2/3 seperately from RWB output. Overlay by custom column
def rwb_groupby_level_nqplot(rwb, ref_ppm=170, overlay_col='IO', single_col_subplot=True, plot_level=-1, xlim=(-1,64)):
    if len(rwb) != rwb[overlay_col].nunique():
        raise Exception('Each row must be a unique value in the groupby column')

    col_names = [x for x in rwb.columns if re.search('^LEVEL_[0-3]_.*ADC_PPM$',x)]
    window_ppm_data = rwb[col_names]

    if plot_level == -1:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 row, 2 columns of subplots
        axs = axs.flatten()
    elif plot_level < 4:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    # Find range of conductance
    cond_list = []
    for col in col_names:
        cond_list.append(int(col.split('_')[2].split('ADC')[0]))
    cond_list = list(set(cond_list))
    cond_list.sort()
    # cond_min = min(cond_list)
    # cond_max = max(cond_list)

    for i in range(len(rwb)):
        # Extract column names (x-axis) and values (y-axis)
        # x = window_ppm_data[[x for x in window_ppm_data if '01' in x]].columns
        x = {} # dict of x values
        y = {} # dict of y values

        x[0] = [0] + cond_list
        y[0] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_0' in x]].iloc[i]
        y[0] = [stats.norm.ppf(1/(64*1296))] + [stats.norm.ppf(x/1e6) for x in y[0]]

        x[1] = cond_list
        y[1] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_1' in x]].iloc[i]
        y[1] = [stats.norm.ppf(x/1e6) for x in y[1]]

        x[2] = cond_list
        y[2] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_2' in x]].iloc[i]
        y[2] = [stats.norm.ppf(x/1e6) for x in y[2]]

        x[3] = cond_list
        y[3] = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_3' in x]].iloc[i]
        y[3] = [stats.norm.ppf(x/1e6) for x in y[3]]

        for level in range(4):
            if np.all(np.abs(y[level])==np.inf):
                x[level] = [63,63]
                y[level] = [stats.norm.ppf((1)/(64*1296)),stats.norm.ppf((64*1296-1)/(64*1296))]

            # If there are np.inf values, pad the ones before the data with lowest sigma and the ones after the data with highest sigma
            elif np.any(np.abs(y[level])==np.inf):
                reached_data = False
                for j in range(len(y[level])):
                    if abs(y[level][j])!=np.inf:
                        reached_data = True

                    if (reached_data==False) and (abs(y[level][j])==np.inf):
                        y[level][j] = stats.norm.ppf((1)/(64*1296))
                    elif reached_data and (abs(y[level][j])==np.inf):
                        y[level][j] = stats.norm.ppf((64*1296-1)/(64*1296))

        if plot_level == -1:
            for level in range(4):
                axs[level].plot(x[level], y[level], marker='o', linestyle='-', label = rwb[overlay_col].values[i])
                axs[level].set_title(f'Level{level}')
                axs[level].set_xlabel("ADC Code")
                axs[level].set_ylabel("Sigma")
                axs[level].set_xlim(xlim)
                axs[level].set_ylim([-4,4])
                axs[level].axhline(stats.norm.ppf(ref_ppm/1e6), color='gray', linestyle='--')
                axs[level].axhline(-stats.norm.ppf(ref_ppm/1e6), color='gray', linestyle='--')
                axs[level].legend()
                axs[level].grid(True)
        else:
            axs.plot(x[plot_level], y[plot_level], marker='o', linestyle='-', label = rwb[overlay_col].values[i])
            axs.set_title(f'Level{plot_level}')
            axs.set_xlabel("ADC Code")
            axs.set_ylabel("Sigma")
            axs.set_xlim(xlim)
            axs.set_ylim([-4,4])
            axs.axhline(stats.norm.ppf(ref_ppm/1e6), color='gray', linestyle='--')
            axs.axhline(-stats.norm.ppf(ref_ppm/1e6), color='gray', linestyle='--')
            axs.legend()
            axs.grid(True)

    plt.tight_layout()
    plt.show()

def rwb_plot_window_ppm(rwb, mean=False):
    col_names = [x for x in rwb.columns if re.search('^LEVEL_(01|12|23)_.*US_PPM$',x)]
    window_ppm_data = rwb[col_names]

    # Find range of conductance
    cond_list = []
    for col in col_names:
        cond_list.append(int(col.split('_')[2].split('US')[0]))
    cond_list = list(set(cond_list))
    cond_list.sort()

    if mean:
        window_ppm_data = window_ppm_data.mean()

        x = cond_list
        y_01 = window_ppm_data.loc[[x for x in col_names if 'LEVEL_01' in x]]
        y_12 = window_ppm_data.loc[[x for x in col_names if 'LEVEL_12' in x]]
        y_23 = window_ppm_data.loc[[x for x in col_names if 'LEVEL_23' in x]]

        plt.plot(x, y_01, marker='o', linestyle='-', label = 'Level 0-1')
        plt.plot(x, y_12, marker='o', linestyle='-', label = 'Level 1-2')
        plt.plot(x, y_23, marker='o', linestyle='-', label = 'Level 2-3')
        plt.title("Mean PPM")
        plt.xlabel("Conductance (uS)")
        plt.ylabel("PPM")
        plt.yscale("log")
        plt.ylim([1e1,1e6])
        plt.legend()
        plt.grid(True)
    else:
        n_rows_to_plot = math.ceil(len(rwb)/2)
        fig, axs = plt.subplots(n_rows_to_plot, 2, figsize=(10, 3*n_rows_to_plot))  # 2 row, 2 columns of subplots
        axs = axs.flatten()

        for i in range(len(rwb)):
            # Extract column names (x-axis) and values (y-axis)
            # x = window_ppm_data[[x for x in window_ppm_data if '01' in x]].columns
            x = cond_list
            y_01 = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_01' in x]].iloc[i]  # Get the first row (since the data is in a single row)
            # print(len(y_01))
            y_12 = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_12' in x]].iloc[i]  # Get the first row (since the data is in a single row)
            # print(len(y_12))
            y_23 = window_ppm_data[[x for x in window_ppm_data if 'LEVEL_23' in x]].iloc[i] # Get the first row (since the data is in a single row)

            # Plot
            # plt.figure(figsize=(8, 6))
            axs[i].plot(x, y_01, marker='o', linestyle='-', label = 'Level 0-1')
            axs[i].plot(x, y_12, marker='o', linestyle='-', label = 'Level 1-2')
            axs[i].plot(x, y_23, marker='o', linestyle='-', label = 'Level 2-3')
            axs[i].set_title(rwb['IO'].values[i])
            axs[i].set_xlabel("Conductance (uS)")
            axs[i].set_ylabel("PPM")
            axs[i].set_yscale("log")
            axs[i].set_ylim([1e1,1e6])
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()

# This function pivots the RWB table by test to facilitate cross-test analysis on the same IOs
def rwb_pivot_by_test(rwb):
    # Define index columns (assumes TEST_NAME is the last index column)
    tstnm_col = list(rwb.columns).index('TEST_NAME')
    index_cols = list(rwb.columns)[:tstnm_col]
    value_cols = list(rwb.columns)[tstnm_col + 1:]

    rwb_split = rwb.pivot(index=index_cols, columns=['TEST_NAME'], values=value_cols).reset_index()
    rwb_split.columns = [' - '.join(col[::-1]).strip() if col[1] != '' else col[0] for col in rwb_split.columns]

    return rwb_split

# This function pivots the RWB table by test to facilitate cross-test analysis on the same IOs
def rwb_pivot_by_test(rwb):
    # Define index columns (assumes TEST_NAME is the last index column)
    tstnm_col = list(rwb.columns).index('TEST_NAME')
    index_cols = ['DIE_ID','MACRO','IO']
    value_cols = list(rwb.columns)[tstnm_col + 1:]

    rwb_split = rwb.pivot(index=index_cols, columns=['TEST_NAME'], values=value_cols).reset_index()
    rwb_split.columns = [' - '.join(col[::-1]).strip() if col[1] != '' else col[0] for col in rwb_split.columns]

    return rwb_split

def rwb_scatter_plot(rwb, x, y, overlay_var, autofit_axis_tol_perc=-1, y_scale = 'linear', fit_type=None):
    # Scatter plot
    plt.figure(figsize=(8, 6))
    unique_overlay_values = rwb[overlay_var].unique()
    
    for var in unique_overlay_values:
        x_data = rwb.loc[rwb[overlay_var] == var][x].values
        y_data = rwb.loc[rwb[overlay_var] == var][y].values
        
        # Scatter plot
        plt.scatter(x_data, y_data, alpha=0.7, label=var)

        if fit_type=='spline':
            # Sort data for spline fitting
            sorted_indices = np.argsort(x_data)
            x_sorted = x_data[sorted_indices]
            y_sorted = y_data[sorted_indices]

            # Fit a Univariate Spline
            spline = spi.UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted))  # s controls smoothing
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)  # Generate smooth x values
            y_smooth = spline(x_smooth)

            # Plot the spline
            plt.plot(x_smooth, y_smooth, linestyle='-') #, label=f'{var} Spline Fit'

        elif fit_type=='linear':
            # Sort data for spline fitting
            sorted_indices = np.argsort(x_data)
            x_sorted = x_data[sorted_indices]
            y_sorted = y_data[sorted_indices]

            # Fit a linear fit
            slope, intercept = np.polyfit(x_sorted, y_sorted, 1)
            y_fit = slope * x_sorted + intercept

            # Plot the fit
            plt.plot(x_sorted, y_fit, label=f"Linear Fit: y = {slope:.2f}x + {intercept:.2f}")

    plt.title(f'{x} vs {y} by {overlay_var}', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.legend()
    
    # Adjust axis limits if required
    if (autofit_axis_tol_perc > 0) and (autofit_axis_tol_perc < 100):
        plt.xlim([np.percentile(rwb[x], autofit_axis_tol_perc), np.percentile(x_data, 100 - autofit_axis_tol_perc)])
        plt.ylim([np.percentile(rwb[y], autofit_axis_tol_perc), np.percentile(y_data, 100 - autofit_axis_tol_perc)])

    plt.yscale(y_scale)  # Keep log scale
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Disable scientific notation
    # plt.gca().ticklabel_format(style='plain', axis='y')  # Enforce plain number format
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def rwb_fetch_data(regex = False, regex_col = None):
    # Define the server and login credentials
    hostname = "192.168.68.215"
    port = 22
    username = "admin2"
    password = "tetra4027"

    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Automatically add the server's host key if it's not already in known_hosts
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        ssh.connect(hostname, port, username, password)

        # Seed the random number generator with the current time
        seed = int(time.time())  # Get the current time as an integer
        # print(f"seed: {seed}")
        random.seed(seed)

        # Loop through rwb db indexes
        df = pd.DataFrame()
        for db_index in [2,3]:
            print(f'Looking at table rwb_db_{db_index}')
            # Generate an 8-digit random number
            random_number = random.randint(10000000, 99999999)
            temp_file_server = f"output_{random_number}.txt"

            # Execute a command
            if regex==False:
                print('Pulling all RWB data, may take awhile')
                stdin, stdout, stderr = ssh.exec_command(f"sudo mysql -D rwb -e 'SELECT * FROM rwb_db_{db_index};' > {temp_file_server}", get_pty=True) # sudo mysql -D rwb -e 'SELECT * FROM rwb_db_2;' > op.txt
            elif regex_col is not None:
                print(f'Fetching {regex} regular expression filter on RWB column {regex_col}')
                stdin, stdout, stderr = ssh.exec_command(f"sudo mysql -D rwb -e \"SELECT * FROM rwb_db_{db_index} WHERE {regex_col} REGEXP '{regex}';\" > {temp_file_server}", get_pty=True)
            else:
                raise Exception('Need to specify a column to filter the regualr expression')
            # stdin, stdout, stderr = ssh.exec_command("sudo mysql -D rwb -e 'SELECT * FROM rwb_db_2;' > op.txt", get_pty=True) # sudo mysql -D rwb -e 'SELECT * FROM rwb_db_2;' > op.txt
            
            # Send password
            stdin.write(password + "\n")
            stdin.flush()

            # Capture and display the command output or errors
            # print("Output:")
            # print(stdout.read().decode())
            # print("Errors:")
            # print(stderr.read().decode())

            # Ensure enough time to pull data
            time.sleep(1)

            # Transfer output file by creating an SFTP client
            sftp = ssh.open_sftp()

            # Transfer the file
            local_rwb_file = 'rwb_pull_temp.txt'
            sftp.get(temp_file_server, local_rwb_file) # temp_file_server
            # print(f"File transferred successfully to {local_rwb_file}")

            # Close the SFTP client
            sftp.close()

            # Return loaded file as dataframe
            try:
                df_local = pd.read_csv(local_rwb_file, delimiter='\t')
                os.remove(local_rwb_file)
            except pd.errors.EmptyDataError:
                print(f'No data pulled from rwb_db_{db_index}')
            else:
                print(f'Data pulled from rwb_db_{db_index}')
                df = pd.concat([df, df_local], ignore_index=True)

            # Delete the temporary file on the server
            stdin, stdout, stderr = ssh.exec_command(f"rm {temp_file_server}")

        print('Pull successful')

    finally:
        # Close the connection
        ssh.close()

    return df

# Function to retrieve DB names
def get_dbnames():
    # Define the server and login credentials
    hostname = "192.168.68.164"
    port = 22
    username = "admin2"
    password = "40271234"

    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Automatically add the server's host key if it's not already in known_hosts
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        ssh.connect(hostname, port, username, password)

        # Seed the random number generator with the current time
        seed = int(time.time())  # Get the current time as an integer
        # print(f"seed: {seed}")
        random.seed(seed)

        # Generate an 8-digit random number
        random_number = random.randint(10000000, 99999999)
        temp_file_server = f"output_databasenames_{random_number}.txt"

        # Get name of all databases to find the correct one:
        stdin, stdout, stderr = ssh.exec_command(f"sudo mysql -e \"SHOW DATABASES;\" > {temp_file_server}", get_pty=True)
        
        # Send password
        stdin.write(password + "\n")
        stdin.flush()

        time.sleep(1)

        # Loop through rwb db indexes
        df = pd.DataFrame()

        # Transfer output file by creating an SFTP client
        sftp = ssh.open_sftp()

        # Transfer the file
        local_dbnames_file = 'databasenames_temp.txt'
        sftp.get(temp_file_server, local_dbnames_file) # temp_file_server
        # print(f"File transferred successfully to {local_rwb_file}")

        # Close the SFTP client
        sftp.close()

        # Return loaded file as dataframe
        df = pd.concat([df, pd.read_csv(local_dbnames_file, delimiter='\t')], ignore_index=True)
        os.remove(local_dbnames_file)

        # Delete the temporary file on the server
        stdin, stdout, stderr = ssh.exec_command(f"rm {temp_file_server}")

        print(' - DB names pull successful')

    finally:
        # Close the connection
        ssh.close()

    return df

# Function to retrieve table names
def get_tablenames(db_name):
    # Define the server and login credentials
    hostname = "192.168.68.164"
    port = 22
    username = "admin2"
    password = "40271234"

    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Automatically add the server's host key if it's not already in known_hosts
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        ssh.connect(hostname, port, username, password)

        # Seed the random number generator with the current time
        seed = int(time.time())  # Get the current time as an integer
        # print(f"seed: {seed}")
        random.seed(seed)

        # Generate an 8-digit random number
        random_number = random.randint(10000000, 99999999)
        temp_file_server = f"output_databasenames_{random_number}.txt"

        # Get name of all databases to find the correct one:
        stdin, stdout, stderr = ssh.exec_command(f"sudo mysql -e \"SHOW TABLES FROM \`{db_name}\`;\" > {temp_file_server}", get_pty=True)
        
        # Send password
        stdin.write(password + "\n")
        stdin.flush()

        time.sleep(1)

        # Loop through rwb db indexes
        df = pd.DataFrame()

        # Transfer output file by creating an SFTP client
        sftp = ssh.open_sftp()

        # Transfer the file
        local_bitlevel_file = 'databasenames_temp.txt'
        sftp.get(temp_file_server, local_bitlevel_file) # temp_file_server
        # print(f"File transferred successfully to {local_rwb_file}")

        # Close the SFTP client
        sftp.close()

        # Return loaded file as dataframe
        df = pd.concat([df, pd.read_csv(local_bitlevel_file, delimiter='\t')], ignore_index=True)

        # Delete the temporary file on the server
        stdin, stdout, stderr = ssh.exec_command(f"rm {temp_file_server}")

        print(' - - Tablename pull successful')

    finally:
        # Close the connection
        ssh.close()

    return df

# This function retrieves the bit-level data for a given db/table
def get_bitleveldata(db_name, table_name):
    # Define the server and login credentials
    hostname = "192.168.68.164"
    port = 22
    username = "admin2"
    password = "40271234"

    # Create an SSH client
    ssh = paramiko.SSHClient()

    # Automatically add the server's host key if it's not already in known_hosts
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        ssh.connect(hostname, port, username, password)

        # Seed the random number generator with the current time
        seed = int(time.time())  # Get the current time as an integer
        # print(f"seed: {seed}")
        random.seed(seed)

        # Loop through rwb db indexes
        df = pd.DataFrame()

        # Generate an 8-digit random number
        random_number = random.randint(10000000, 99999999)
        temp_file_server = f"output_bitlevel_{random_number}.txt"

        # Execute a command
        stdin, stdout, stderr = ssh.exec_command(f"sudo mysql -D \"{db_name}\" -e 'SELECT * FROM `{table_name}`;' > {temp_file_server}", get_pty=True) # sudo mysql -D rwb -e 'SELECT * FROM rwb_db_2;' > op.txt
        # stdin, stdout, stderr = ssh.exec_command("sudo mysql -D rwb -e 'SELECT * FROM rwb_db_2;' > op.txt", get_pty=True) # sudo mysql -D rwb -e 'SELECT * FROM rwb_db_2;' > op.txt
        
        # Send password
        stdin.write(password + "\n")
        stdin.flush()

        # Capture and display the command output or errors
        # print("Output:")
        # print(stdout.read().decode())
        # print("Errors:")
        # print(stderr.read().decode())

        # Ensure enough time to pull data
        time.sleep(1)

        # Transfer output file by creating an SFTP client
        sftp = ssh.open_sftp()

        # Transfer the file
        local_bitlevel_file = 'bitlevel_pull_temp.txt'
        sftp.get(temp_file_server, local_bitlevel_file) # temp_file_server
        # print(f"File transferred successfully to {local_rwb_file}")

        # Close the SFTP client
        sftp.close()

        # Return loaded file as dataframe
        df = pd.concat([df, pd.read_csv(local_bitlevel_file, delimiter='\t')], ignore_index=True)

        # Delete the temporary file on the server
        stdin, stdout, stderr = ssh.exec_command(f"rm {temp_file_server}")

        print(' - - - Bit-level pull successful')

    finally:
        # Close the connection
        ssh.close()

    return df

# Takes in a dictionary of bit-level 2-d numpy bit maps and outputs into a stacked dataframe
def bitlevel_npydict_to_df(npy_file_dict, pattern='rowbar'):
    df_stacked = pd.DataFrame()
    length = len(npy_file_dict)
    i = 0
    for key, data in npy_file_dict.items():
        print(f" - - - Flattening {i+1}/{length} npy files")

        # Get the array's shape
        rows, cols = data.shape

        # Create the DataFrame
        if rows > 64:
            df_temp = pd.DataFrame({
                'WL': np.repeat(np.arange(rows), cols),
                'BL': np.tile(np.arange(cols), rows),
                'ADC': data.flatten(),
                'TEST': key
            })
        else:
            df_temp = pd.DataFrame({
                'BL': np.repeat(np.arange(rows), cols),
                'WL': np.tile(np.arange(cols), rows),
                'ADC': data.flatten(),
                'TEST': key
            })
        
        # Include level information
        print(f' - - - Pattern used for level deciphering is {pattern}')
        if pattern == 'rowbar':
            df_temp['LEVEL'] = (df_temp.WL/324).astype(int)

        elif pattern == 'pr1':
            # Upload sub sample 64 BLs x 162 WLs pattern
            pr1 = np.loadtxt(r"pseudorandom_mlm_pattern_1.csv", delimiter=',', dtype=int)
            pr1_rows, pr1_cols = pr1.shape
            pr1_stacked_sub = pd.DataFrame({
                'BL': np.repeat(np.arange(pr1_rows), pr1_cols),
                'WL': np.tile(np.arange(pr1_cols), pr1_rows),
                'LEVEL': pr1.flatten()
            })

            # Expand to full 1296 WLs then merge with bit-level data
            pr1_stacked = pd.DataFrame()
            for j in range(8):
                pr1_stacked_sub['WL'] = pr1_stacked_sub.WL + 162
                pr1_stacked = pd.concat([pr1_stacked, pr1_stacked_sub])
            df_temp['LEVEL'] = pd.merge(left=df_temp, right=pr1_stacked, on=['BL','WL'])

        df_stacked = pd.concat([df_stacked, df_temp], ignore_index=True)

        i += 1

    # Split df by 'TEST'
    df = df_stacked.pivot(values=['ADC'], columns=['TEST'], index=['BL','WL','LEVEL']).reset_index()
    
    # Flatten columns by joining levels with '_'
    df.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in df.columns]
    
    return df

# This function uses iterates through each row in an RWB dataframe and fetches the bit-level data.
# The fetch db names, fetch table names, fetch bit-level data and npy-to-df post process functions are used in here
def rwb_fetch_bitlevel_data(rwb, default_pattern='rowbar'):
    # Fetch database names
    print('Getting database names')
    df_db_names = get_dbnames()
    
    # Bit-level dataframe to return at the end of the function
    df = pd.DataFrame()

    # Loop through each row in rwb and get the data
    rwb.reset_index(inplace=True)

    # Define pattern if no PATTERN column
    if 'PATTERN' not in rwb.columns:
        rwb['PATTERN'] = default_pattern

    if len(rwb) > 20:
        raise Warning(f'Length of raw readouts to pull is high ({len(rwb)}), recommend putting less than 20 entried to save time')

    for i, row in rwb.iterrows():
        print(f'Fetching row {i+1} of {len(rwb)} from input RWB table')
        # Get meta data from each rwb row
        DIE_ID = row.DIE_ID
        MACRO = row.MACRO
        IO = row.IO
        RUN_NAME = row.RUN_NAME
        TEST_NAME = row.TEST_NAME
        PATTERN = row.PATTERN
        TEST_START_DATETIME = row.TEST_START_DATETIME.replace('_','').replace('-','')

        # Find matching db names
        df_matching_dbs = df_db_names.loc[(df_db_names.Database.str.contains(RUN_NAME))&(df_db_names.Database.str.contains(f'{DIE_ID}-{MACRO}'))]
        print(f' - Found {len(df_matching_dbs)} matching databases')
        
        # Search for matching output files in each db
        matching_db_table = []
        for j, db in df_matching_dbs.iterrows():
            db_name = db.Database
            print(f' - Database {j}: {db_name}')

            df_tables_names = get_tablenames(db_name)
            df_tables_names.columns = ['Table']
            df_matching_tables = df_tables_names.loc[(df_tables_names.Table.str.contains(f'_io{IO}_'))&(df_tables_names.Table.str.contains(f'{TEST_NAME}_adc'))]

            for k, table_name in df_matching_tables.iterrows():
                print(f' - - Table {k}: {table_name.Table}')
                matching_db_table.append({'db_name': db_name, 'table_name': table_name.Table})
            
        if len(matching_db_table) > 1:
            raise Warning(f'Error: {len(matching_db_table)} tables found when expecting one, not importing bit-level data')
        elif len(matching_db_table) == 0:
            raise Warning('Error: no tables found')
        else:
            df_temp = get_bitleveldata(db_name=matching_db_table[0]['db_name'], table_name=matching_db_table[0]['table_name'])
            
            temp_data = {}
            temp_data[f'{RUN_NAME}_{TEST_NAME}_{TEST_START_DATETIME}'] = df_temp.to_numpy()
            df_temp_postproc = bitlevel_npydict_to_df(temp_data, PATTERN)
            df_temp_postproc.insert(0, 'DIE_ID', DIE_ID)
            df_temp_postproc.insert(1, 'MACRO', MACRO)
            df_temp_postproc.insert(2, 'IO', IO)
            
            if df.empty:
                df = df_temp_postproc.copy()
            else:
                df = pd.merge(left=df, right=df_temp_postproc, how='outer', on=['DIE_ID','MACRO','IO','BL','WL','LEVEL'])

                # Identify the column names with _x and _y
                for col in df.columns:
                    if col.endswith('_x'):
                        base_col = col[:-2]  # Remove '_x' suffix
                        col_y = f"{base_col}_y"
                        if col_y in df.columns:
                            df[base_col] = df[col].combine_first(df[col_y])  # Merge data from both columns
                            df.drop([col, col_y], axis=1, inplace=True)  # Drop old columns

    return df