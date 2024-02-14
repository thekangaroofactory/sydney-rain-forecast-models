
# ---------------------------------------------------------------------------------------
# Australia, Rain Tomorrow.
# ---------------------------------------------------------------------------------------

# -- 1. Introduction
# Dataset is provided by the Australian Government - Bureau of Meteorology (BOM): http://www.bom.gov.au/climate/data/

# It contains 140000+ examples, captured in different locations accross Australia, with daily values and label
# whether there was rain or not on the next day.
# Goal is to predict whether it will rain on the next day.

# Imports
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt


# dataset path
raw_dataset_url = '/kaggle/input/australia_rain_tomorrow_raw.csv'


# -- 2. Quick data observation

# First, let's capture basic information about the dataset (available features, types, number of examples).

# Read column names from file
feature_list = list(pd.read_csv(raw_dataset_url, nrows =1))

print(feature_list)

# The 'RISK_MM' feature is a prediction of the amout of rain for the next day (in mm), so this has to be ignored.

# list of columns to remove
list_remove = ['RISK_MM']

# define headers to load
header_list = [feature for feature in feature_list if feature not in list_remove]

# open dataset csv file with desired columns
raw_dataset = pd.read_csv(raw_dataset_url, sep = ',', usecols = header_list)

# Let's print a sample of the dataset examples.
print(raw_dataset.sample(5))

# get number of examples (rows)
nb_row = raw_dataset.shape[0]
print('Number of examples:', nb_row)

# get No/Yes ratio for 'RainTomorrow'
raw_dataset['RainTomorrow'].value_counts()

# That's going to be our reference: setting all predictions to 'No' gives 77.58% accuracy.


# -- 3. NaN values analysis

# init list
nan_report = [None] * len(header_list)

# loop over feature list
for index, feature in enumerate(header_list):
    nan_report[index] = raw_dataset[feature].isna().sum()
    print(feature, ': ', nan_report[index], ' / ', round(nan_report[index] / nb_row * 100, 2), '%')


# -- 3.1 Drop features

# Features with huge amount of NaN values will be dropped from dataset:
# - Evaporation :  42.79 %
# - Sunshine :  47.69 %
# - Cloud9am :  37.74 %
# - Cloud3pm :  40.15 %

# list of features to be dropped
drop_list = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm']

# drop those features from raw_dataset
raw_dataset.drop(columns=drop_list, inplace=True)

# update lists -- remove dropped features
nan_report = [value for value, feature in zip(nan_report, header_list) if feature not in drop_list]
header_list = [feature for feature in header_list if feature not in drop_list]

# check
print(raw_dataset.sample(5))


# -- 3.2 Extract list of numerical and categorical features

# init lists
numerical_feature = []
categorical_feature = []

# loop over dataframe types
for feature, type in enumerate(raw_dataset.dtypes):
    if type == 'float64': # assuming I checked raw_dataset.dtypes before
        numerical_feature.append(header_list[feature])
    else: # non 'float64' are objects
        categorical_feature.append(header_list[feature])

print('Numerical features: ', numerical_feature)
print('\nCategorical features: ', categorical_feature)


# -- 3.3 Categorical features

# From here I will make a copy of raw_dataset as eng_dataset to keep the raw data safe if needed later.

eng_dataset = raw_dataset.copy()


# Okay let's have a look at the NaN status for our categorical features:

# loop over categorical_feature
for feature in categorical_feature:
    index = header_list.index(feature)
    print(feature, ': ', nan_report[index], ' / ', round(nan_report[index] / nb_row * 100, 2), '%')


# - 'Date', 'Location' and 'RainTomorrow' are all good, nothing to do :)
# - 'RainToday' is missing when previous day is not part of the dataset. Since that's 0.99% of the examples,
# I'm just going to put 'No' where it's missing - it would be interresting to see what happens if you delete
# those rows or set them to some 'unknown' value. (but not expecting a huge impact out of 0.99%)
# - 'WindGustDir', 'WindDir9am' and 'WindDir3pm' are wind directions (N, NNE, NE, ...),
# I'm adding 'UNK' for unknown ; you might want to try to fill NaN values with maybe the most common value
# per Location (I didn't try), or even maybe to just delete those rows as well.

# fill 'RainToday' NaN values with 'No'
eng_dataset['RainToday'].fillna('No', inplace=True)

# fill 'WindGustDir', 'WindDir9am' and 'WindDir3pm' NaN values with 'UNK'
for feature in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
    eng_dataset[feature].fillna('UNK', inplace=True)


# -- 3.4 Numerical features

# loop over numerical_feature
for feature in numerical_feature:
    index = header_list.index(feature)
    print(feature, ': ', nan_report[index], ' / ', round(nan_report[index] / nb_row * 100, 2), '%')


# I'm going to replace NaN by mean values, but in order to be more specific,
# I want to capture something like means by location by whether it will rain tomorrow or not.

# init columns
columns = numerical_feature.copy()
columns.insert(0, 'Location')

# create empty dataframes with shape
mean_by_location_rain = {}
mean_by_location_norain = {}


# define utility to get means by location
def mean_by_location(input_df, rain):
    
    output_dict = {}
    working_df = input_df.loc[input_df['RainTomorrow'] == rain].copy()
    
    # loop over input_df group by 'Location'
    for location, databyloc_df in working_df[columns].groupby('Location'):
    
        loc_mean = {}
    
        # loop over columns of databyloc_df
        for feature in list(databyloc_df[numerical_feature]):

            loc_mean[feature] = databyloc_df[feature].mean()
            
        # append local dict to output_list
        output_dict[location] = loc_mean

    return(output_dict)


# call utility
mean_by_location_rain = mean_by_location(eng_dataset, rain='Yes')
mean_by_location_norain = mean_by_location(eng_dataset, rain='No')


# There are some location/feature without any value:
# Albany has no WindGustSpeed value at all for example.
# One option could be to delete those rows again, but I will first try to fill those ones with global means
# over all locations depending on if it will rain tomorrow or not (that means WindGustSpeed for Albany will
# receive a different mean value for RainTomorrow=Yes and RainTomorrow=No)

# Let's build dictionaries to store global means (one if rain, one if no rain):

# init
mean_by_rain = {}
mean_by_norain = {}

# loop over numerical_feature
for feature in numerical_feature:
    mean_by_rain[feature] = eng_dataset.loc[eng_dataset['RainTomorrow'] == 'Yes', feature].mean()
    mean_by_norain[feature] = eng_dataset.loc[eng_dataset['RainTomorrow'] == 'No', feature].mean()
    
print('mean_by_rain: ', mean_by_rain)
print('\nmean_by_norain: ', mean_by_norain)

# fill local NaN values based on global means
mean_by_location_rain = {location: {feature: mean_by_rain.get(feature) if math.isnan(value) else value for feature, value in loc_dict.items()}
                         for location, loc_dict in mean_by_location_rain.items()}

mean_by_location_norain = {location: {feature: mean_by_norain.get(feature) if math.isnan(value) else value for feature, value in loc_dict.items()}
                         for location, loc_dict in mean_by_location_norain.items()}


# Now let's fill the numerical NaN values by:
# - first choice: mean by location by whether it will rain tomorrow or not,
# - if not available: mean by whether it will rain tomorrow or not.

print('Nb of NaN:\n', eng_dataset[numerical_feature].isna().sum())


# loop over location_list
for (location, rain_dict), (_, norain_dict) in zip(mean_by_location_rain.items(), mean_by_location_norain.items()):
    
    mask = (eng_dataset['Location'] == location) & (eng_dataset['RainTomorrow'] == 'Yes')
    eng_dataset.loc[mask] = eng_dataset.loc[mask].fillna(value=rain_dict)

    mask2 = (eng_dataset['Location'] == location) & (eng_dataset['RainTomorrow'] == 'No')
    eng_dataset.loc[mask2] = eng_dataset.loc[mask2].fillna(value=norain_dict)

print('\nNb of NaN:\n', eng_dataset[numerical_feature].isna().sum())


# -- 4. Feature engineering

print(eng_dataset.sample(5))


# -- 4.1 Date feature

# Date values are way too specific. I decided to just extract a new 'Month' feature from it,
# in order to introduce some seasonality.
# At this point, I don't believe days are very useful, maybe week in month (1,2,3,4) or first/second half
# of the month could help but I didn't try.
# I will skip year as well, feel free to try and see if it helps or not.

# Split Date into Day, Month, Year and drop Date, Day, Year
eng_dataset[['Day','Month', 'Year']] = eng_dataset['Date'].str.split("/", expand=True)
eng_dataset.drop(['Date', 'Day', 'Year'], axis=1, inplace=True)

print(eng_dataset.sample(2))


# *Note: I obviously missed to convert Month values to something acceptable by TensorFlow...
# You'll find a hack later on to fix that*

# -- 4.2 Location

# There are 49 locations in the dataset. That's a lot of different categories for a single feature.
# Some locations are very similar: Sydney and SydneyAirport or Melbourne and Richmond are very similar
# locations as compared to the size of Australia!
# Actually I decided to group locations by climate type after a look at the climate zone:

# ![](https://greenharvest.com.au/Images/Miscellaneous/AustralianClimateZoneMap.png)

# So Cairns is going to be 'Tropical', Sydney is 'Temperate', Perth is 'SubTropical' and Mildura is 'Grassland'...

# create location mapping dictionnary (climate types are already indexed)
location_mapping = {'Adelaide': 0, 'Albany': 0, 'Albury': 0, 'AliceSprings': 1,
                    'BadgerysCreek': 0, 'Ballarat': 0, 'Bendigo': 0, 'Brisbane': 2,
                    'Cairns': 3, 'Canberra': 0, 'Cobar': 4, 'CoffsHarbour': 2,
                    'Dartmoor': 0, 'Darwin': 3, 'GoldCoast': 2, 'Hobart': 0, 'Katherine': 3,
                    'Launceston': 0, 'Melbourne': 0, 'MelbourneAirport': 0, 'Mildura': 4,
                    'Moree': 2, 'MountGambier': 0, 'MountGinini': 0, 'Newcastle': 0, 'Nhil': 4,
                    'NorahHead': 0, 'NorfolkIsland': 2, 'Nuriootpa': 0, 'PearceRAAF': 2,
                    'Penrith': 0, 'Perth': 2, 'PerthAirport': 2, 'Portland': 0, 'Richmond': 0,
                    'Sale': 0, 'SalmonGums': 4, 'Sydney': 0, 'SydneyAirport': 0, 'Townsville': 3,
                    'Tuggeranong': 0, 'Uluru': 1, 'WaggaWagga': 0, 'Walpole': 0, 'Watsonia': 0,
                    'Williamtown': 0, 'Witchcliffe': 0, 'Wollongong': 0, 'Woomera': 1}


# replace location by climate type index
eng_dataset['Location'].replace(location_mapping, inplace=True)

print(eng_dataset.sample(5))


# -- Other categorical features

# Let's index all other categorical features (WindGustDir, WindDir9am, WindDir3pm, RainToday and RainTomorrow).

# extract unique values (they are same for WindGustDir, WindDir9am and WindDir3pm)
wind_unique_values = eng_dataset['WindGustDir'].unique()
wind_unique_values.sort()

# create wind dir mapping
wind_mapping = {key: value for value, key in enumerate(wind_unique_values)}

print('wind_mapping =', wind_mapping)

# create yes/no mapping
binary_mapping = {'No': 0, 'Yes':1}

print('binary_mapping =', binary_mapping)

# replace
eng_dataset[['WindGustDir', 'WindDir9am', 'WindDir3pm']] = eng_dataset[['WindGustDir', 'WindDir9am', 'WindDir3pm']].replace(wind_mapping)
eng_dataset[['RainToday', 'RainTomorrow']] = eng_dataset[['RainToday', 'RainTomorrow']].replace(binary_mapping)

print(eng_dataset.sample(5))


# -- 5. Feature normalization

# Okay now we're going to normalize all numerical features - not the categorical ones - with (x - mean / range).
# Feel free to try various normalization approaches and see how it impacts the accuracy.

# init
mean_dict = {}
range_dict = {}

# loop over numerical features
for feature in numerical_feature:
    
    # compute means
    mean_dict[feature] = eng_dataset[feature].mean()
    
    # compute range
    range_dict[feature] = eng_dataset[feature].max() - eng_dataset[feature].min()
    
    # nomalize feature
    eng_dataset[feature] = (eng_dataset[feature] - mean_dict[feature]) / range_dict[feature]

    
print('mean_dict:', mean_dict)
print('\nrange_dict:', range_dict)

print(eng_dataset.sample(5))


# *Time to fixe the Month type issue... (TensorFlow will throw an ambigous error at model.fit() time otherwise)*

# fixing month type..
eng_dataset['Month'] = eng_dataset['Month'].astype(float)


# # 6. Model definition and training

# -- 6.1 Parameters

# Label column
label_col = ['RainTomorrow']

# Dataset split
split = [0.80, 0.10, 0.10]
shuffle = True

# model
learning_rate = 0.001
batch_size = 32
epochs = 500 # I did 100 / 300 before, just trying to see where we go
steps_per_epoch = 10


# -- 6.2 Shuffle and split dataset

# utility to shuffle and split dataset
def split_dataset(input_df, label_col, ratio_list, shuffle):
    
    # Parameters:
    # -----------
    #    - input_df: input dataframe
    #    - label_col: name of the label column in input_df
    #    - ratio_list: [] of ratios = 0.x
    #    - shuffle: boolean if shuffle is requested
    # Output:
    # -------
    #    - training_df, validation_df, test_df
    
    temp_df = input_df.copy()
    
    # Shuffle dataset
    if (shuffle):
        print('Shuffle dataset')
        temp_df.sample(frac=1)

    # Extract and drop labels from dataset
    labels = temp_df[label_col].copy()
    temp_df = temp_df.drop(label_col, axis=1)

    # Compute split indexes
    nb_row = temp_df.shape[0]
    print('nb_row =', nb_row)
    index_1 = int(nb_row * split[0])
    index_2 = int(nb_row * (split[0] + split[1]))
    print('index_1 =', index_1)
    print('index_2 =', index_2)

    # Split
    training_set = temp_df[:index_1]
    validation_set = temp_df[index_1:index_2]
    test_set = temp_df[index_2:]
    training_label = labels[:index_1]
    validation_label = labels[index_1:index_2]
    test_label = labels[index_2:]

    # Check
    print ('\nTraining set :', training_set.shape)
    print ('Training labels :', training_label.shape)
    print ('Validation set :', validation_set.shape)
    print ('Validation labels :', validation_label.shape)
    print ('Test set :', test_set.shape)
    print ('Test labels :', test_label.shape)
    
    # return
    return training_set, training_label, validation_set, validation_label, test_set, test_label


# call split utility
training_set, training_label, validation_set, validation_label, test_set, test_label = split_dataset(eng_dataset, label_col, split, shuffle)


# -- 6.3 Model definition

# clean session
tf.keras.backend.clear_session()

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', dtype='float32'),
    tf.keras.layers.Dense(128, activation='relu', dtype='float32'),
    tf.keras.layers.Dense(64, activation='relu', dtype='float32'),
    tf.keras.layers.Dense(32, activation='relu', dtype='float32'),
    tf.keras.layers.Dense(16, activation='relu', dtype='float32'),
    tf.keras.layers.Dense(8, activation='relu', dtype='float32'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# define compile options
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics='binary_accuracy')


# -- 6.4 Model training

# train model
history = model.fit(
    x=training_set,
    y=training_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(validation_set, validation_label),
    steps_per_epoch=steps_per_epoch,
)

# accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# -- 6.5 Make predictions

# make prediction
raw_predictions = model.predict(
    test_set,
    batch_size=None,
    verbose=1,
)


# utility to replace by binary value
def make_binary(input_data, threshold):
    
    output = input_data.copy()
    output = np.nan_to_num(output)
    output[output > threshold] = 1
    output[output <= threshold] = 0
    
    return output


# apply
binary_predictions = make_binary(raw_predictions, 0.43)
predictions_df = pd.DataFrame(binary_predictions)
predictions_df.rename({0: label_col[0]}, axis=1, inplace=True)
print(predictions_df.sample(5))


# -- 6.6 Validate predictions

# init
nb_test_example = test_set.shape[0]
index = list(range(0, nb_test_example))
metrics = pd.DataFrame(np.nan, index=index, columns=['true_positive', 'false_positive', 'false_negative', 'true_negative'])

# reindex labels
test_label.reset_index(drop=True, inplace=True)

# compare predictions with labels 
compare = list((predictions_df == test_label).any(1))

# get count of prediction OK
prediction_ok = compare.count(True)
prediction_ko = compare.count(False)

# compute accuracy
accuracy = prediction_ok / test_set.shape[0]

# compare predictions with labels
metrics['true_positive'] = np.where((predictions_df['RainTomorrow'] == True) & (test_label['RainTomorrow'] == True), True, False)
metrics['false_positive'] = np.where((predictions_df['RainTomorrow'] == True) & (test_label['RainTomorrow'] == False), True, False)
metrics['false_negative'] = np.where((predictions_df['RainTomorrow'] == False) & (test_label['RainTomorrow'] == True), True, False)
metrics['true_negative'] = np.where((predictions_df['RainTomorrow'] == False) & (test_label['RainTomorrow'] == False), True, False)

# extract counts
true_positive = np.count_nonzero(metrics['true_positive'])
false_positive = np.count_nonzero(metrics['false_positive'])
false_negative = np.count_nonzero(metrics['false_negative'])
true_negative = np.count_nonzero(metrics['true_negative'])

# Precision, recall
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

# F1Scrore
f1_score = (2 * precision * recall) / (precision + recall)

# display
print('nb_row test set :', test_set.shape[0])

print('\nPrediction OK :', prediction_ok)
print('Prediction KO :', prediction_ko)
print('Accuracy :', round(accuracy * 100, 2), '%')

print('\nTrue_positive :', true_positive)
print('False_positive :', false_positive)
print('False_negative :', false_negative)
print('True_negative :', true_negative)

print('\nPrecision :', round(precision, 2))
print('Recall :', round(recall, 2))

print('\nF1 Score:', round(f1_score, 2))


# -- 7. Conclusion

# Actually the 88% accuracy is better than my draft version
# (was ~87%, I did a bit of engineering and fixed some issues).
# I would be interesting to try different options for NaN values, Location, Date,
# Wind directions - I didn't do so far - and check if it improves the predictions.

# My feeling is that since the dataset contains 110316 examples without rain tomorrow and
# 31877 with rain (this is Australia ^^),
# the model struggles a bit to correctly predict for rain:
# * False negative are quite high (an analysis of this could help improve the model)
# * Best result is obtained with a threshold at 0.43.. so it's like the model predictions need
# a little boost! (0.50 resulted in 86.xx% accuracy)
