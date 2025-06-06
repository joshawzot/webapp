# Data Range Filter Feature

## Overview
The Data Range Filter feature allows users to filter data points to only include values within a specified numerical range during plot generation and analysis. This feature helps focus analysis on specific data ranges and exclude outliers or irrelevant data points.

## Features Added

### 1. Form Input Fields
- **Minimum Value**: Optional field to set the lower bound for data filtering
- **Maximum Value**: Optional field to set the upper bound for data filtering
- Both fields support decimal numbers and can be left empty
- Fields are located in the "Data Range Filter" section of the input form

### 2. Form Validation
- **Range Validation**: Ensures minimum value is less than maximum value when both are provided
- **Client-side Validation**: JavaScript validation prevents form submission with invalid ranges
- **Error Messages**: Clear error messages guide users to correct invalid inputs

### 3. Backend Processing
- **Data Filtering**: Applied during data matrix processing in `generate_plot.py`
- **NaN Handling**: Filtered-out data points are replaced with NaN values
- **Robust Processing**: All plotting functions handle NaN values correctly
- **Debug Logging**: Console output shows filtering progress and valid data counts

### 4. Visual Feedback
- **Information Banner**: Results page displays an alert banner when data range filtering is applied
- **Filter Details**: Shows the exact range used for filtering
- **Visual Distinction**: Uses warning-style alert to clearly indicate data has been filtered

## Usage Instructions

### Basic Usage
1. Navigate to the plot generation form
2. Scroll to the "Data Range Filter (optional)" section
3. Enter desired minimum and/or maximum values
4. Submit the form to generate filtered plots

### Filter Options
- **Both Min and Max**: `5` to `15` - includes only data points between 5 and 15
- **Minimum Only**: `≥ 10` - includes only data points 10 and above
- **Maximum Only**: `≤ 8` - includes only data points 8 and below
- **No Filter**: Leave both fields empty to include all data points

### Validation Rules
- If both minimum and maximum values are provided, minimum must be less than maximum
- Equal values (min = max) are not allowed
- Only one value can be provided without validation error
- Empty fields are ignored (no filtering applied)

## Technical Implementation

### Files Modified
1. **`templates/input_form_generate_plot.html`**
   - Added data range input fields
   - Added client-side validation JavaScript
   - Added form reset functionality for new fields

2. **`route_handlers.py`**
   - Added data range field processing in `get_form_data_generate_plot()`
   - Added template variables for displaying filter information
   - Added error handling for invalid input conversion

3. **`generate_plot.py`**
   - Added data filtering logic in the main processing loop
   - Modified matrix-based functions to handle NaN values
   - Added `calculate_ber_with_target_ranges()` function
   - Added debug logging for filter progress

4. **`templates/plot.html`**
   - Added information banner for data range filter status
   - Displays filter parameters and helpful information

### Data Processing Flow
1. **Input Processing**: Form data converted to float values or None
2. **Matrix Filtering**: Boolean mask applied to data matrices
3. **NaN Replacement**: Filtered-out values replaced with NaN
4. **Group Processing**: Functions filter out NaN values during calculations
5. **Plot Generation**: Plotting functions handle sparse data correctly

## Testing
A test script `test_data_range_filter.py` is included to verify:
- Data filtering logic with various range combinations
- Form validation behavior
- NaN handling in data processing

Run tests with:
```bash
python3 test_data_range_filter.py
```

## Benefits
- **Focused Analysis**: Analyze specific data ranges without manual preprocessing
- **Outlier Exclusion**: Easily exclude extreme values that might skew results
- **Flexible Filtering**: Support for one-sided or two-sided range filtering
- **User-Friendly**: Intuitive interface with clear validation and feedback
- **Robust Implementation**: Handles edge cases and maintains data integrity

## Future Enhancements
- **Percentile-based Filtering**: Filter based on data percentiles
- **Multiple Range Support**: Support for multiple non-contiguous ranges
- **Statistical Summaries**: Show statistics of filtered vs. original data
- **Filter Presets**: Save and reuse common filter configurations 