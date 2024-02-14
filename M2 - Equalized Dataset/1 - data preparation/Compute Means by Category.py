
# ---------------------------------------------------------------------------------------
# Compute Means by Category
# Description: Computes counts, sums, means of dedicated columns based on given column category
# ---------------------------------------------------------------------------------------

# -- Imports

# Imports
import pandas as pd
import math
import numpy as np


# -- Input parameters

# csv file name
csv_filename = 'equalized_dataset_raw.csv'

# output csv file name
output_csv = 'means_by_location.csv'

# chunk size
chunk_size = 10000

category_col = ['Location']
target_col = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']
header_list = category_col + target_col


# -- Open CSV file

# Open csv file with desired columns
df_chunk = pd.read_csv(csv_filename, sep = ',', chunksize = chunk_size, usecols = header_list)


# -- Computes Means based on Category

# Init output
output_Location = []
output_Count = []
output_SumEvaporation = []
output_CountEvaporation = []
output_SumSunshine = []
output_CountSunshine = []
output_SumCloud9am = []
output_CountCloud9am = []
output_SumCloud3pm = []
output_CountCloud3pm = []

# Loop over chunk
print('\nStart looping over chunks:')

for chunk in df_chunk:
    
    # get start stop index
    start = chunk.index.min()
    stop = chunk.index.max()
    
    print('-- Chunk : start = ' + str(start) + ', stop = ' + str(stop))

    for _, grp_df in chunk.groupby('Location'):
        
        location = grp_df['Location'].unique()[0]
        
        # append Location to output if not yet in
        if location not in output_Location:
            print('Adding new location to list:', location)
            output_Location.append(location)
            output_Count.append(0)
            output_SumEvaporation.append(0)
            output_CountEvaporation.append(0)
            output_SumSunshine.append(0)
            output_CountSunshine.append(0)
            output_SumCloud9am.append(0)
            output_CountCloud9am.append(0)
            output_SumCloud3pm.append(0)
            output_CountCloud3pm.append(0)

        # get location index
        index = output_Location.index(location)
        
        # update output count and sums
        for _, row in grp_df.iterrows():
            output_Count[index] += 1
            if not math.isnan(row['Evaporation']):
                output_CountEvaporation[index] += 1
                output_SumEvaporation[index] += row['Evaporation']
            if not math.isnan(row['Sunshine']):
                output_CountSunshine[index] += 1
                output_SumSunshine[index] += row['Sunshine']
            if not math.isnan(row['Cloud9am']):
                output_CountCloud9am[index] += 1
                output_SumCloud9am[index] += row['Cloud9am']
            if not math.isnan(row['Cloud3pm']):
                output_CountCloud3pm[index] += 1
                output_SumCloud3pm[index] += row['Cloud3pm']

# Check output
print(output_Location)
print(output_Count)

# Convert lists to np.arrays
np_Count = np.array(output_Count)
np_SumEvaporation = np.array(output_SumEvaporation)
np_CountEvaporation = np.array(output_CountEvaporation)
np_SumSunshine = np.array(output_SumSunshine)
np_CountSunshine = np.array(output_CountSunshine)
np_SumCloud9am = np.array(output_SumCloud9am)
np_CountCloud9am = np.array(output_CountCloud9am)
np_SumCloud3pm = np.array(output_SumCloud3pm)
np_CountCloud3pm = np.array(output_CountCloud3pm)

# Compute means
np_MeanEvaporation = np_SumEvaporation / np_CountEvaporation
np_MeanSunshine = np_SumSunshine / np_CountSunshine
np_MeanCloud9am = np_SumCloud9am / np_CountCloud9am
np_MeanCloud3pm = np_SumCloud3pm / np_CountCloud3pm

# print
print(np_MeanEvaporation)
print(np_MeanSunshine)
print(np_MeanCloud9am)
print(np_MeanCloud3pm)


# -- Build dataframe and save as CSV

# merge all
np_output = np.column_stack((output_Location, np_Count,
                            np_SumEvaporation, np_CountEvaporation, np_MeanEvaporation,
                            np_SumSunshine, np_CountSunshine, np_MeanSunshine,
                            np_SumCloud9am, np_CountCloud9am, np_MeanCloud9am,
                            np_SumCloud3pm, np_CountCloud3pm, np_MeanCloud3pm
                      ))

# build dataframe
output_df = pd.DataFrame(np_output)
col_dict = {0: "Location", 1: "Count",
            2: "SumEvaporation", 3: "CountEvaporation", 4: "MeanEvaporation",
            5: "SumSunshine", 6: "CountSunshine", 7: "MeanSunshine",
            8: "SumCloud9am", 9: "CountCloud9am", 10: "MeanCloud9am",
            11: "SumCloud3pm", 12: "CountCloud3pm", 13: "MeanCloud3pm"
           }
output_df.rename(columns=col_dict, inplace=True)

print(output_df)


# -- Save Dataframe to CSV

# save
output_df.to_csv(output_csv, index=False, chunksize=chunk_size)
