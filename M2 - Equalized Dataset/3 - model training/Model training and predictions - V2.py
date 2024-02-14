
# ---------------------------------------------------------------------------------------
# Model training and predictions
# Description:
#
#    - Split dataset into training / validation / test sets
#    - Define and train model
#    - Make predictions and validate
# ---------------------------------------------------------------------------------------

# -- Imports
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# -- Input parameters

# Dataset csv file name
dataset_url = 'processed_dataset__v1.csv'

# List of column to remove
list_remove = ['RISK_MM', 'Cloud9am', 'Cloud3pm', 'Evaporation', 'Sunshine']

# Label column
label_col = ['RainTomorrow']

# Dataset split
split = [0.80, 0.10, 0.10]
shuffle = True


# -- Load dataset

# Read column names from file
header_list = list(pd.read_csv(dataset_url, nrows =1))

# Remove unwanted columns from header list
header_list = [i for i in header_list if (i not in list_remove)]

# Open csv file with desired columns
dataset_df = pd.read_csv(dataset_url, sep = ',', usecols = header_list)

# Check
print('Dataset size :', dataset_df.shape)
dataset_df.sample(5)


# -- Split dataset into training / validation / test

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
training_set, training_label, validation_set, validation_label, test_set, test_label = split_dataset(dataset_df, label_col, split, shuffle)


# -- Model definition

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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# define compile options
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])


# -- Train model

# train model
history = model.fit(
    x=training_set.values,
    y=training_label.values,
    batch_size=32,
    epochs=500,
    verbose=1,
    validation_data=(validation_set, validation_label),
    steps_per_epoch=10,
)


# -- Plot results

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


# -- Make predictions

# make prediction
raw_predictions = model.predict(
    test_set,
    batch_size=None,
    verbose=1,
)


# replace by binary value
def make_binary(input_data, threshold):
    output = input_data.copy()
    output = np.nan_to_num(output)
    output[output > threshold] = 1
    output[output <= threshold] = 0
    
    return output


# apply
binary_predictions = make_binary(raw_predictions, 0.50)
predictions_df = pd.DataFrame(binary_predictions)
predictions_df.rename({0: label_col[0]}, axis=1, inplace=True)

predictions_df.sample(5)


# -- Validate predictions

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


print(metrics['false_negative'])


# -- Export test result

# create dataframe and reindex
export_df = test_set.copy()
export_df.set_index(test_label.index, inplace=True)

# merge labels and predictions
export_df['RainTomorrow'] = test_label
export_df['Prediction'] = predictions_df

# save dataframe to csv
export_df.to_csv('test_predictions.csv', mode='w', header=True, index=False)


# -- Save model

# Save the entire model to a HDF5 file
model.save('rain_tomorrow_v2_equalized.h5')

print(dataset_df.dtypes)
