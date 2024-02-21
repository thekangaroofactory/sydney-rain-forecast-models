
# ---------------------------------------------------------------------------------------
# Feature Engineering
# Description:
#
#    - Perform feature engineering (add, merge columns...)
#    - Fill NaN values based on strategies
#
# ---------------------------------------------------------------------------------------

# -- Imports

import pandas as pd
import json
import math


# -- Input parameters

# Raw dataset csv file name
# raw_dataset_url = 'weather_aus_raw.csv'
raw_dataset_url = '../4 - recurrent check/formated_data.csv'

# output processed dataset csv file name
processed_dataset_url = '../4 - recurrent check/processed_dataset_v1.csv'


# JSON report file name
json_report_url = '../outputs/resources/dataset_report.json'

# Mean by location csv file name
mean_by_location_url = '../outputs/resources/means_by_location.csv'

# Categorical feature bucket mapping
bucket_mapping = True
bucket_features = ['Location']

# chunk size
chunk_size = 10000

# list of column to remove
list_remove = ['RISK_MM']

# list of NaN default values
categorical_nan = ['None', 'UNK', 'UNK', 'UNK', 'No', 'None']


# -- Load JSON report & extract lists

# Load JSON file
with open(json_report_url) as json_file:
    data = json.load(json_file)
    
# Get number of rows and drop key from dict
nb_row = data.pop('nb_row', None)

# Create dataframe from dict
report_df = pd.DataFrame.from_dict(data, orient='columns', dtype=None, columns=None)

print('\nNumber of row :', nb_row)
print(report_df)

# Extract lists from report
feature_list = report_df['colname'].tolist()

# Extract numerical related lists
numerical_features = [colname for colname, coltype in zip(report_df['colname'], report_df['coltype']) if coltype == 'float']
range_list = [rg for rg, coltype in zip(report_df['range'], report_df['coltype']) if coltype == 'float']
mean_list = [mean for mean, coltype in zip(report_df['mean'], report_df['coltype']) if coltype == 'float']

# Extract categorical related lists
categorical_features = [colname for colname, coltype in zip(report_df['colname'], report_df['coltype']) if coltype == 'str']
category_list = [unique for unique, coltype in zip(report_df['unique_str'], report_df['coltype']) if coltype == 'str']

# build header list
header_list = ['Date'] + feature_list

# check
print('\n', feature_list)
print('\n', numerical_features)
print('\n', range_list)
print('\n', mean_list)
print('\n', categorical_features)
print('\n', category_list)


# -- Load bucket mappings

# check activation
if bucket_mapping:
    
    mapping_list = []
    
    # loop over feature list
    for feature in bucket_features:
        
        # open corresponding mapping_feature.json file
        json_mapping_url = 'E:/Portfolio/Python/Projects/sydney-rain-forecast-models/M1 - Complete Dataset/outputs/resources/mapping_' + feature + '.json'
        with open(json_mapping_url) as json_file:
            data = json.load(json_file)
        mapping_list.append(data)
        
print(mapping_list)


# -- Build categorical mappings

# init
categorical_mapping = [None] * len(categorical_features)

# loop over categorical features
for index, values in enumerate(category_list):
    
    # add nan default to list
    if categorical_nan[index] != 'None' and categorical_nan[index] not in values:
        values.insert(0, categorical_nan[index])
            
    # build mapping list
    categorical_mapping[index] = { values[i] : i for i in range(0, len(values)) }

# check
print(categorical_mapping)


# -- Load mean by location report

# Open csv file
mean_by_loc_df = pd.read_csv(mean_by_location_url, sep = ',')

print(mean_by_loc_df)


# -- Open CSV file

# Open csv file with desired columns
df_chunk = pd.read_csv(raw_dataset_url, sep = ',', chunksize = chunk_size, usecols = header_list)


# -- Feature engineering

# Loop over chunk
print('\nStart looping over chunks:')

# init csv params
header = True
mode = 'w'
            
for chunk in df_chunk:
    
    # get start stop index
    start = chunk.index.min()
    stop = chunk.index.max()
        
    print('-- Chunk : start = ' + str(start) + ', stop = ' + str(stop))

    # loop over location group
    for _, grp_df in chunk.groupby('Location'):
        
        # get location
        location = grp_df['Location'].unique()[0]
        print('   Location = ', location)
    
        # copy local input
        output = grp_df.copy()

        # Step.1: Split Date into Day, Month, Year and drop Date, Day, Year
        # output[['Day', 'Month', 'Year']] = output['Date'].str.split("/", expand=True)
        output[['Day', 'Month', 'Year']] = output['Date'].str.split("-", expand=True)
        output.drop(['Date', 'Day', 'Year'], axis=1, inplace=True)
    
    
        # Step.2: Categorical features
        for index, col in enumerate(categorical_features):
    
            # 2.1: Fill NaN values 
            output.loc[output[col].isnull(), col] = categorical_nan[index]
          
            # 2.2: Index categorical features
            if bucket_mapping and col in bucket_features:
                idx = bucket_features.index(col)
                output[col].replace(mapping_list[idx], inplace=True) 
            else:
                output[col].replace(categorical_mapping[index], inplace=True)

                
        # Step.3: Numerical features
        # get location index in mean_by_loc_df
        loc_index = mean_by_loc_df.index[mean_by_loc_df['Location'] == location]
        # loop over numerical_features
        for index, col in enumerate(numerical_features):
        
            # 3.1: Fill NaN values 
            # get default mean from global means
            value = mean_list[index]
        
            # col name to look for
            meancol = 'Mean' + col
            # check if the mean column exists in mean_by_loc_df
            if meancol in mean_by_loc_df.columns:
                # check if mean column has value
                if not math.isnan(mean_by_loc_df[meancol][loc_index]):
                    # update value with local mean
                    value = mean_by_loc_df[meancol][loc_index].item()

            output.loc[output[col].isnull(), col] = value
    
            # 3.2: features normalization [(x - mean) / range]
            output[col] = output[col].apply(lambda x: (x - mean_list[index])/range_list[index])
        

        # Step.4: write dataframe into file
        print('   >> Saving updated chunk to csv file')
        output.to_csv(processed_dataset_url, mode=mode, header=header, index=False)
        # update csv params to switch to append mode
        if header:
            header = False
            mode = 'a'

            
# Check (last chunk)
print(output.sample(5))
