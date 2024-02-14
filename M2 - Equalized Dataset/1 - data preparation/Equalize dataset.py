
# ---------------------------------------------------------------------------------------
# Equalize dataset
# Description:
#
#    - Check amount of Positive / Negative examples
#    - Pick random examples of the minority group to equalize both
#
# ---------------------------------------------------------------------------------------

# -- Imports

# Imports
import pandas as pd


# -- Input parameters

# Raw dataset csv file name
raw_dataset_url = 'weather_aus_raw.csv'

# Output file name
output_url = 'equalized_dataset_raw.csv'


# -- Load dataset

# load csv into dataframe
data_df = pd.read_csv(raw_dataset_url, sep=',')


# -- Split positive / negative examples

# split
positive_df = data_df[data_df.RainTomorrow == 'Yes']
negative_df = data_df[data_df.RainTomorrow == 'No']

# check dims
print('   Total nb examples = ', data_df.shape[0])
print('   Nb positive examples = ', positive_df.shape[0])
print('   Nb negative examples = ', negative_df.shape[0])

# select direction
max_examples = min(positive_df.shape[0], negative_df.shape[0])

# subset
if negative_df.shape[0] > max_examples:
    subset_negative_df = negative_df.sample(n=max_examples)
    subset_positive_df = positive_df
else:
    subset_negative_df = negative_df
    subset_positive_df = positive_df.sample(n=max_examples)

# check size
print('   Nb subset positive examples = ', subset_positive_df.shape[0])
print('   Nb subset negative examples = ', subset_negative_df.shape[0])

# merge
new_dataset = pd.concat([subset_positive_df, subset_negative_df], axis=0)
print('   Nb new_dataset examples = ', new_dataset.shape[0])

# randomize
new_dataset = new_dataset.sample(frac=1)

# check
print(new_dataset)


# -- Save

# Step.4: -- write dataframe into file
print('>> Saving output to csv file')
output.to_csv(processed_dataset_url, mode=mode, header=header, index=False)
