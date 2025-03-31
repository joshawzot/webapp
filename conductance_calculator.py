"""
Conductance Calculator Module for the webapp.
This module provides functions to calculate conductance values based on input parameters.
"""

import numpy as np
import pandas as pd

# Predefined lookup tables for FLINT calculations
flint_lookup = {
    "BLDRV_Bl_R": {
        0: 100.0,
        1: 150.0,
        2: 200.0,
        3: 250.0,
        4: 300.0,
        5: 350.0,
        6: 400.0,
        7: 450.0
    },
    "Bleed_R": {
        0: 50.0,
        1: 100.0,
        2: 150.0,
        3: 200.0,
        4: 250.0, 
        5: 300.0,
        6: 350.0,
        7: 400.0
    }
}

# Configuration for the conductance calculator
flint_conductance_calculator = {
    "REF_Scaling_Factor": 1.0,
    "Typical_BL_RD_IN_resistance": 30.0  # in kOhm
}

# Linear conversion constants
LINEAR_CONVERSION = {
    "input_min": 0,
    "input_max": 63,
    "output_min": 60,
    "output_max": 170
}

def run_flint_conductance_calculator(input_params):
    """
    Calculate conductance based on the input parameters.
    
    Args:
        input_params (dict): A dictionary containing the input parameters for the calculation.
    
    Returns:
        float: The calculated conductance value in microSiemens (uS).
    """
    # Calculate intermediate and output values
    real_VCM_BUF = input_params["Observed_VCM_BUF_on_UGB33"] - input_params["UGB33_offset"]
    real_VREF_BUF = input_params["Observed_VREF_BUF_on_UGB3"] - input_params["UGB33_offset"]
    real_BL = input_params["Observed_BL_FB_on_UGB11"] - input_params["UGB11_offset"]
    real_1p1_LDO = input_params["Observed_1p1V_LDO_output_on_UGB33"] - input_params["UGB33_offset"]
    
    Vout_for_ADC_output_0 = real_VCM_BUF - real_VREF_BUF / 2 / flint_conductance_calculator["REF_Scaling_Factor"]
    ADC_mv_per_LSB = real_VREF_BUF / 63 / flint_conductance_calculator["REF_Scaling_Factor"]
    
    ideal_ADC_output = input_params["Observed_ADC_Output_code"] - input_params["ADC_offset"]
    predicted_Vout = Vout_for_ADC_output_0 + (63 - ideal_ADC_output) * ADC_mv_per_LSB
    
    BLDRV_BL_R = flint_lookup["BLDRV_Bl_R"][input_params["Setting_for_BLDRV_BL_R"]]
    total_current = (real_1p1_LDO - predicted_Vout) / BLDRV_BL_R
    
    bleeder_R = flint_lookup["Bleed_R"][input_params["Setting_for_BLDRV_BLEED_RD"]]
    bleeder_G = 1000 / bleeder_R  # because unit for resistance is kohm here.
    
    full_path_G = (
        (total_current / (real_BL / 1000)) - bleeder_G
    ) / (
        1
        + flint_conductance_calculator["Typical_BL_RD_IN_resistance"]
        * bleeder_G
        / 1000
    )
    
    # Convert to microSiemens (uS)
    conductance_uS = full_path_G * 1e6
    
    return conductance_uS

def convert_table_to_conductance(table_data, input_params):
    """
    Convert all values in a table to conductance values.
    
    Args:
        table_data (numpy.ndarray): The original table data.
        input_params (dict): The input parameters for the conductance calculation.
    
    Returns:
        numpy.ndarray: The converted table data with conductance values.
    """
    # Make a copy of the input parameters to avoid modifying the original
    conductance_table = np.zeros_like(table_data, dtype=float)
    
    # Iterate through the table and convert each value
    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            params = input_params.copy()
            params["Observed_ADC_Output_code"] = float(table_data[i, j])
            conductance_table[i, j] = run_flint_conductance_calculator(params)
    
    return conductance_table

def get_unique_original_values_and_conductance(table_data, input_params):
    """
    Get a list of unique original values and their corresponding conductance values.
    
    Args:
        table_data (numpy.ndarray): The original table data.
        input_params (dict): The input parameters for the conductance calculation.
    
    Returns:
        list: A list of dictionaries containing original and conductance values.
    """
    # Get unique values from the table
    unique_values = np.unique(table_data)
    comparison = []
    
    # Calculate conductance for each unique value
    for value in unique_values:
        params = input_params.copy()
        params["Observed_ADC_Output_code"] = float(value)
        conductance = run_flint_conductance_calculator(params)
        comparison.append({
            "original": float(value),
            "conductance": conductance
        })
    
    return comparison

def run_linear_conversion(value):
    """
    Perform a linear conversion from input range (0-63) to output range (60-170).
    
    Args:
        value (float): The original value to convert.
    
    Returns:
        float: The linearly converted value.
    """
    # Extract conversion parameters
    input_min = LINEAR_CONVERSION["input_min"]
    input_max = LINEAR_CONVERSION["input_max"]
    output_min = LINEAR_CONVERSION["output_min"]
    output_max = LINEAR_CONVERSION["output_max"]
    
    # Linear mapping formula: output = output_min + (value - input_min) * (output_max - output_min) / (input_max - input_min)
    # Ensure value is within the input range
    clamped_value = max(input_min, min(value, input_max))
    
    # Calculate the linear conversion
    converted = output_min + (clamped_value - input_min) * (output_max - output_min) / (input_max - input_min)
    
    return converted

def convert_table_to_linear(table_data):
    """
    Convert all values in a table using linear conversion.
    
    Args:
        table_data (numpy.ndarray): The original table data.
    
    Returns:
        numpy.ndarray: The converted table data.
    """
    # Create a new array with the same shape
    converted_table = np.zeros_like(table_data, dtype=float)
    
    # Apply linear conversion to each element
    for i in range(table_data.shape[0]):
        for j in range(table_data.shape[1]):
            converted_table[i, j] = run_linear_conversion(float(table_data[i, j]))
    
    return converted_table

def get_unique_original_values_and_linear_conversion(table_data):
    """
    Get a list of unique original values and their corresponding linearly converted values.
    
    Args:
        table_data (numpy.ndarray): The original table data.
    
    Returns:
        list: A list of dictionaries containing original and converted values.
    """
    # Get unique values from the table
    unique_values = np.unique(table_data)
    comparison = []
    
    # Calculate converted value for each unique original value
    for value in unique_values:
        converted = run_linear_conversion(float(value))
        comparison.append({
            "original": float(value),
            "converted": converted
        })
    
    return comparison 