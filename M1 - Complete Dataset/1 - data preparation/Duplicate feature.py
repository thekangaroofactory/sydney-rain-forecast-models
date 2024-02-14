
# ---------------------------------------------------------------------------------------
# Duplicate Feature
# Description: Perform feature duplication (square, cube, square root)
# ---------------------------------------------------------------------------------------

# -- Imports

# Imports
import pandas as pd


# -- Input parameters

# Raw dataset csv file name
raw_dataset_url = 'weather_aus_raw.csv'

# Categorical feature bucket mapping
feature_list = {
    'MaxTemp': ['square', 'cube', 'sqrt'],
    'Rainfall': ['square', 'cube', 'sqrt'],
    'Humidity3pm': ['square', 'cube', 'sqrt']
}

# chunk size
chunk_size = 10000

# output processed dataset csv file name
processed_dataset_url = 'weather_aus_polyfeature_raw.csv'


# -- Open CSV file

# Open csv file with desired columns
df_chunk = pd.read_csv(raw_dataset_url, sep=',', chunksize=chunk_size)


# -- Feature duplication

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

    # local copy
    output = chunk.copy()
    
    # loop over dictionnary
    for key, value in feature_list.items():
        print('Treating feature :', key)
        
        # loop over operations
        for operation in value:
            print('   - operation :', operation)
            
            # compute feature name
            new_key = key + '_' + operation
            
            # case square operation
            if operation == 'square':
                output[new_key] = output[key]**2
            # case cube operation
            if operation == 'cube':
                output[new_key] = output[key]**3
            # case square root operation
            if operation == 'sqrt':
                output[new_key] = output[key]**(1/2)

    # write dataframe into file
    print('   >> Saving updated chunk to csv file')
    output.to_csv(processed_dataset_url, mode=mode, header=header, index=False)
    # update csv params to switch to append mode
    if header:
        header = False
        mode = 'a'    
