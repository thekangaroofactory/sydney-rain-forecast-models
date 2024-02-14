
# ---------------------------------------------------------------------------------------
# JSON Report Display
# Description: Loads and displays the JSON data_analysis_report file.
# ---------------------------------------------------------------------------------------

# -- Imports
import pandas as pd
import json


# -- Input parameters

# JSON file path and name
json_url = 'dataset_report.json'


# -- Load JSON file

# Load JSON file
with open(json_url) as json_file:
    data = json.load(json_file)


# -- Build dataframe from dict

# Get number of rows and drop key from dict
nb_row = data.pop('nb_row', None)

# Create dataframe from dict
report_df = pd.DataFrame.from_dict(data, orient='columns', dtype=None, columns=None)


# -- Display report
print('\n nb_row :', nb_row)
print('\n Report :')

print(report_df)
