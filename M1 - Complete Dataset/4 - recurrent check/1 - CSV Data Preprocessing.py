
# ---------------------------------------------------------------------------------------
# CSV Data Preprocessing
# Description:
#
#    - Perform preprocessing of raw .csv data
#    - Skip unwanted lines
#    - Rename columns to fit with original raw data
#    - Add missing columns to fit with original raw data
#
# Input data:
#    - CSV file from http://www.bom.gov.au/climate/dwo/[yyyymm]/text/[Location].[yyyymm].csv
#
# ---------------------------------------------------------------------------------------

# Next versions:
# - Update Location column based on file name [Location].[yyyymm].csv with external dict mapping file

# -- Imports
from os import listdir, path, rename
import pandas as pd
import numpy as np


# -- Declare parameters

# folder path
input_path = 'dataset/incoming'
output_path = 'dataset/preprocessed'
archive_path = 'dataset/raw_archive'

# file name
output_file = 'formated_data.csv'


# -- 1. List files in folder

# list files in input folder
file_list = listdir(input_path)

print('Files to be processed: ', len(file_list))
print(file_list)


# -- 2. Process files

# Process files in list
raw_df_list = []
for file in file_list:
    
    print('   Processing file: ', file)
    
    # build target url
    target_url = path.join(input_path, file)
    
    # Open csv file (ignore line starting with " and blank lines)
    raw_df = pd.read_csv(target_url, sep=',', comment='"', skip_blank_lines=True, encoding='ANSI')
    
    # Remove unwanted columns
    raw_df.drop(columns=['Unnamed: 0', 'Time of maximum wind gust'], inplace=True)
    
    # Rename columns to fit original data
    raw_df.rename(columns={'Date':'Date',
                           'Minimum temperature (°C)':'MinTemp',
                           'Maximum temperature (°C)':'MaxTemp',
                           'Rainfall (mm)':'Rainfall',
                           'Evaporation (mm)':'Evaporation',
                           'Sunshine (hours)':'Sunshine',
                           'Direction of maximum wind gust ':'WindGustDir',
                           'Speed of maximum wind gust (km/h)':'WindGustSpeed',
                           '9am Temperature (°C)':'Temp9am',
                           '9am relative humidity (%)':'Humidity9am',
                           '9am cloud amount (oktas)':'Cloud9am',
                           '9am wind direction':'WindDir9am',
                           '9am wind speed (km/h)':'WindSpeed9am',
                           '9am MSL pressure (hPa)':'Pressure9am',
                           '3pm Temperature (°C)':'Temp3pm',
                           '3pm relative humidity (%)':'Humidity3pm',
                           '3pm cloud amount (oktas)':'Cloud3pm',
                           '3pm wind direction':'WindDir3pm',
                           '3pm wind speed (km/h)':'WindSpeed3pm',
                           '3pm MSL pressure (hPa)':'Pressure3pm'},
                     inplace=True)
    
    # Reorder columns
    col_list = ['Date',
                'MinTemp',
                'MaxTemp',
                'Rainfall',
                'Evaporation',
                'Sunshine',
                'WindGustDir',
                'WindGustSpeed',
                'WindDir9am',
                'WindDir3pm',
                'WindSpeed9am',
                'WindSpeed3pm',
                'Humidity9am',
                'Humidity3pm',
                'Pressure9am',
                'Pressure3pm',
                'Cloud9am',
                'Cloud3pm',
                'Temp9am',
                'Temp3pm']
    raw_df = raw_df[col_list]
    
    # Cast columns to expected types
    # Numerical columns (non numerical will become NA)
    numcol_list = ['MinTemp',
                   'MaxTemp',
                   'Rainfall',
                   'Evaporation',
                   'Sunshine',
                   'WindGustSpeed',
                   'WindSpeed9am',
                   'WindSpeed3pm',
                   'Humidity9am',
                   'Humidity3pm',
                   'Pressure9am',
                   'Pressure3pm',
                   'Cloud9am',
                   'Cloud3pm',
                   'Temp9am',
                   'Temp3pm']
    raw_df[numcol_list] = raw_df[numcol_list].apply(pd.to_numeric, errors='coerce')
    # String columns
    strcol_list = ['WindGustDir',
                   'WindDir9am',
                   'WindDir3pm']
    raw_df[strcol_list] = raw_df[strcol_list].astype(str)
    # Date column
    raw_df['Date'] = pd.to_datetime(raw_df['Date'],format = '%Y-%m-%d')
    
    # Replace " " values by NA
    raw_df.replace(r'^\s*$', np.nan, regex = True, inplace = True)
    
    # Add columns to fit original data
    raw_df.insert(loc = 0, column = 'Location', value = "Sydney") # check file name IDCJDW2124.202008.csv to get location mapping
    raw_df['RainToday'] = ['Yes' if x > 0 else 'No' for x in raw_df['Rainfall']]
    raw_df['RISK_MM'] = 0 # will be ignored anyway
    raw_df['RainTomorrow'] = "NA" # default value
    
    # store raw_df in list
    raw_df_list.append(raw_df)
    
    # Move input file to archive
    old_path_file = target_url
    new_path_file = path.join(archive_path, file)
    rename(old_path_file, new_path_file)


# -- 3. Merge and save

# Merge df in list to output
output_df = pd.concat(raw_df_list, ignore_index = True)

# Fill in RainTomorrow values (last raw remains as "-")
for i in range(0, len(output_df)-1):
    output_df.loc[i, 'RainTomorrow'] = output_df.loc[i+1, 'RainToday']
    
# Save ouput to file (append if exists)
output_csv = path.join(output_path, output_file)
header = not path.exists(output_csv)
output_df.to_csv(output_csv, index = False, mode = 'a', header = header)


# -- 4. Check types

# Column types
colt = {'Date': str,
        'Location': str,
        'MinTemp': float,
        'MaxTemp': float,
        'Rainfall': float,
        'Evaporation': float,
        'Sunshine': float,
        'WindGustDir': str,
        'WindGustSpeed': float,
        'WindDir9am': str,
        'WindDir3pm': str,
        'WindSpeed9am': float,
        'WindSpeed3pm': float,
        'Humidity9am': float,
        'Humidity3pm': float,
        'Pressure9am': float,
        'Pressure3pm': float,
        'Cloud9am': float,
        'Cloud3pm': float,
        'Temp9am': float,
        'Temp3pm': float,
        'RainToday': str,
        'RainTomorrow': str}

# load
test_df = pd.read_csv(output_csv, sep = ',', dtype = colt, parse_dates = ['Date'])

# print
print(test_df)
print(test_df.dtypes)
print(test_df.isnull().sum())
