import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import os as os

def create_sequences(dataset, seq_length):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(dataset[i][:seq_length - 1])
        y.append(dataset[i][seq_length - 1:])
    # for i in range(len(dataset) - seq_length):
    #    x.append(dataset[i:i + seq_length])
    #    y.append(dataset[i + seq_length])
    return np.array(x), np.array(y)


# load the data
file = "data/DB/train_DB.csv"
df = pd.read_csv(file)

data_header = [  # 'distanza_da_centralina_cm',
    'year', 'month',
    'week',
    'day_of_year', 'day_of_month', 'day_of_week',
    'hour',
    'complete_timestamp(YYYY_M_DD_HH_M)',
    # 'barometer_hpa', 'temp_c', 'hightemp_c', 'lowtemp_c',
    # 'hum', 'dewpoint__c', 'wetbulb_c',
    # 'windspeed_km_h',
    # 'windrun_km',
    # 'highwindspeed_km_h',
    # 'windchill_c',
    # 'heatindex_c', 'thwindex_c', 'thswindex_c', 'rain_mm', 'rain_rate_mm_h',
    # 'solar_rad_w_m_2', 'solar_energy_ly', 'high_solar_rad_w_m_2', 'ET_Mm',
    # 'heating_degree_days', 'cooling_degree_days', 'humidity_rh',
    # 'solar_klux',
    'temp_centr',
    'temp_to_estimate',
    'sensor'
]

index = data_header.index('complete_timestamp(YYYY_M_DD_HH_M)')
df.dropna(inplace=True)

# Assuming 'df' is your DataFrame and 'temp_to_estimate' and 'temp2' are the columns to be normalized
columns_to_scale = ['temp_to_estimate', 'temp_centr']

# Create a dictionary to store individual scalers for each column
scalers = {}

# Normalize each column separately and store the scaler in the 'scalers' dictionary
for column in columns_to_scale:
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    scalers[column] = scaler

data = df[data_header].values
for i in range(len(data)):
    data[i][index] = data[i][index][len(data[i][index]) - 1]



data = np.asarray(data).astype(np.float32)

print('Shape of data: ', data.shape)

# reshape the data: the input of the LSTM model is a 3D array of shape = (samples, time.steps, features ) where
# samples = the number of time-series (or sequences)
# time-steps = the number of instants in each time-series
# features = the number of elements in each item of the sequence
x, y = create_sequences(data, len(data_header))

# print('Shape of x: ', x.shape)
# print('Shape of y: ', y.shape)

# reshape => dim1 = # samples = # rows / # time-steps, dim2 = # time-steps, dim3 = #variables in each timestep
timesteps = 3
# check if the dimension are compatible!
samples = int(x.shape[0] / timesteps)
x = x[:samples * timesteps]
y = y[:samples * timesteps]
variables = x.shape[1]
x = x.reshape(samples, timesteps, variables)
y = y.reshape(samples, timesteps, 1)

# print('Shape of x after reshape: ', x.shape)

# split data into training and testing set
# train_size = int(samples * 0.8)
# x_train, x_test = x[:train_size], x[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

print('Shape of train data: ', x_train.shape)
print('Shape of train values: ', y_train.shape)
# print('Shape of test data: ', x_test.shape)
# print('Shape of test values: ', y_test.shape)
# print('Shape of test values: ', y[train_size:].shape)


# build the model
# return_sequences = True

model = Sequential()
# input_shape = #timesteps x #variables
model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['mape','mae','mse'])

# train the model
dropout = 0.55
epochs = 300
batch_size = 10
validation_split = 0

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)


plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['mape'], label='mape')
# plt.plot(history.history['mae'], label='mae')
# plt.plot(history.history['mse'], label='mse')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''
# Visualizzo i risultati dell'addestramento
test_results = {}
test_results = model.evaluate(X_test, y_test)
print(test_results)
'''

model.save('data/DB/temperature_prediction_model.h5')

model = load_model('data/DB/temperature_prediction_model.h5') 
test_results = model.evaluate(x_test, y_test)
print('Test results: ', test_results)

y_pred = model.predict(x_test)
target_column = 'temp_to_estimate'  # Replace with the actual column name
# y_pred = y_pred.reshape(-1, 1)
# denormalized_predictions = scalers[target_column].inverse_transform(y_pred)

# # Extract the actual values from the original test set
# actual_values = scalers[target_column].inverse_transform(y_test.reshape(-1, 1))

# # Combine the predicted and actual values for comparison
# comparison_df = pd.DataFrame({'Predicted': denormalized_predictions.flatten(), 'Actual': actual_values.flatten()})
# print(comparison_df)

# # Plot the comparison
# plt.figure(figsize=(10, 6))
# plt.plot(comparison_df['Predicted'], label='Predicted')
# plt.plot(comparison_df['Actual'], label='Actual')
# plt.title('Comparison of Predicted and Actual Values')
# plt.xlabel('Time Steps')
# plt.ylabel('Temperature')
# plt.legend()
# plt.show()
# test_results = model.evaluate(x_test, y_test)
# print(test_results)
y_pred = model.predict(x_test)
print( "Prediction, real value, central value")
for i in range(len(y_pred)):
    #max = max(data[index])
    #min = min(data[index])
    #y_pred_s = (y_pred[i] * (max-min)) + min
    #y_test_s = (y_test[i][2] * (max-min)) + min
    y_pred_temp = y_pred[i].reshape(-1, 1)
    denormalized_predictions = scalers[target_column].inverse_transform(y_pred_temp)
    
    y_test_temp = y_test[i][timesteps-1].reshape(-1, 1)
    denormalized_test = scalers[target_column].inverse_transform(y_test_temp)
    
    print( denormalized_predictions, ",", denormalized_test)
