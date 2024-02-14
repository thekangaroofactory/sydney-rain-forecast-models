
# ---------------------------------------------------------------------------------------
# Dataset JSON Report Builder
# ---------------------------------------------------------------------------------------
#
# Description:
#
#    - Computes the following list of parameters:
#        * 'nb_row': Number of rows in the dataset
#        * 'colname': Names of the columns
#        * 'coltype': Types of the columns
#        * 'nb_nan': Number of NaN values
#        * 'ratio_nan': Ratio of NaN values (xx,xx% of nb_row)
#        * 'unique_str': List of unique strings for string type columns
#        * 'range': Range of values for numerical columns
#        * 'mean': Mean values for numerical columns
#
#    - Saves report into JSON file
#
# ---------------------------------------------------------------------------------------

# -- Imports

import pandas as pd
import math
import json


# -- Input parameters

# csv file path and name
working_directory = '../../datasets'
filename = 'weather_aus_raw'
dataset_url = working_directory + '\\' + filename + '.csv'

# chunk size
chunk_size = 10000

# list of column to remove
list_remove = ['Date', 'RISK_MM']


# -- Manage column list

# Read column names from file
header_list = list(pd.read_csv(dataset_url, nrows=1))

# Remove unwanted columns from header list
header_list = [i for i in header_list if (i not in list_remove)]

print('\nHeader list after remove:')
print(header_list)


# -- Open CSV file

# Open csv file with desired columns
df_chunk = pd.read_csv(dataset_url, sep=',', chunksize=chunk_size, usecols=header_list)


# -- NaN, Mean, Ranges and Single values data analysis

# Init lists
list_col_types = {key: [] for key in header_list}
list_unique_values = {key: [] for key in header_list}
list_nb_nan = {key: 0 for key in header_list}
list_sum = {key: 0 for key in header_list}
list_min = {key: float("inf") for key in header_list}
list_max = {key: 0 for key in header_list}

# Loop over chunk
print('\nStart looping over chunks:')
stop = 0
for chunk in df_chunk:
    
    # perform data filtering 
    start = chunk.index.min()
    stop = chunk.index.max()
    
    print('-- Chunk : start = ' + str(start) + ', stop = ' + str(stop))

    for col, key in zip(list(chunk), list_unique_values):
        
        for value in chunk[col]:
            
            if isinstance(value, float):
                
                if math.isnan(value):
                    list_nb_nan[key] += 1
                else:
                    if not list_col_types[key]:
                        list_col_types[key] = 'float'
                    list_sum[key] += value                    
                    if value < list_min[key]:
                        list_min[key] = value
                    if value > list_max[key]:
                        list_max[key] = value 
                    
            else:
                
                if isinstance(value, str):
                    if not list_col_types[key]:
                        list_col_types[key] = 'str'
                    if value not in list_unique_values.get(key):
                        list_unique_values[key].append(value)

# store max row number
nb_row = stop + 1

# sort unique values
for key in list_unique_values:
    list_unique_values[key].sort()

# compute ranges
difference = []
for imax, imin in zip(list_max.values(), list_min.values()):
    difference.append(imax - imin)

# compute means
mean = []
for isum in list_sum.values():
    mean.append(isum / nb_row)

# compute NaN ratio
nan_ratio = list({k: round(v / nb_row * 100, 2) for k, v in list_nb_nan.items()}.values())

# display analysis results

print('\nHeader list:', header_list)
print('\nColumn types:', list(list_col_types.values()))
print('\nNaN values:', list(list_nb_nan.values()))
print('\nNaN ratio:', list(nan_ratio))
print('\nUnique values:', list(list_unique_values.values()))
print('\nRanges:', difference)
print('\nMeans:', mean)
print('\nNumber of rows:', nb_row)


# ### Prepare JSON content and save file

# prepare dict to save
json_dict = {'nb_row': nb_row, 'colname': header_list, 'coltype': list(list_col_types.values()),
             'nb_nan': list(list_nb_nan.values()), 'ratio_nan': nan_ratio,
             'unique_str': list(list_unique_values.values()), 'range': difference, 'mean': mean}

# save dict as json file
json_export_file = working_directory + 'dataset_report.json'
with open(json_export_file, 'w') as fp:
    json.dump(json_dict, fp, indent=4)
