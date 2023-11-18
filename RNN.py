import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Function to create sequences
def create_sequences(dataset, seq_length):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(dataset[i][:seq_length - 1])
        y.append(dataset[i][seq_length - 1:])
    return np.array(x), np.array(y)

def create_model():
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
    return model


train = True
normalize_flag = True
dropout = 0.55
epochs = 500
batch_size = 32
validation_split = 0.2
timesteps = 3

if train:
    # Data preprocessing
    data_header = [
        # 'sensor', 
        'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
        'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
    ]
    
    
    
    # Load the data
    file = "data/DB/train_DB.csv"
    df = pd.read_csv(file)
    index = data_header.index('complete_timestamp(YYYY_M_DD_HH_M)')
    df.dropna(inplace=True)
    
    
    # Normalize data
    scalers = {}
    if normalize_flag:
        columns_to_scale = ['temp_to_estimate','temp_centr']
        
        for column in columns_to_scale:
            scaler = MinMaxScaler(feature_range=(0,1))
            df[column] = scaler.fit_transform(df[[column]])
            scalers[column] = scaler

    
   
    data = df[data_header].values
    for i in range(len(data)):
        data[i][index] = data[i][index][len(data[i][index]) - 1]
    data = np.asarray(data).astype(np.float32)
    
    
    
    
        
        
    
    
    x, y = create_sequences(data, len(data_header))
    
    # Reshape the data
    samples = int(x.shape[0] / timesteps)
    x = x[:samples * timesteps].reshape(samples, timesteps, x.shape[1])
    y = y[:samples * timesteps].reshape(samples, timesteps, 1)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)



    # Build the model
    model = create_model()
    model.summary()
    checkpointer = ModelCheckpoint(filepath = 'data/DB/temperature_prediction_model.hdf5', verbose = 2, save_best_only = True)
    # Train the model       
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks = [checkpointer])
    
    # Visualize the training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save the trained model
    
    trained_data = dict()
    trained_data['x_train']=x_train
    trained_data['x_test']=x_test
    trained_data['y_train']=y_train
    trained_data['y_test']=y_test
    trained_data['scalers']=scalers
    
    with open('data/trained_data', 'wb') as fout:
        pickle.dump(trained_data, fout)
    fout.close()
    
    
else:
# Load the trained model for testing
    model = load_model('data/DB/temperature_prediction_model.hdf5')
    fin_name = 'data/states_dict'
    with open('data/trained_data', 'rb') as fin:
        trained_data = pickle.load(fin)
    fin.close()
    
    x_train = trained_data['x_train']
    x_test = trained_data['x_test']
    y_train = trained_data['y_train']
    y_test = trained_data['y_test']
    scalers = trained_data['scalers']
    
# Evaluate the model on the test set
test_results = model.evaluate(x_test, y_test)
print('Test results: ', test_results)

# Make predictions and denormalize
y_pred = model.predict(x_test)
print("Prediction, real value, central value")

for i in range(len(y_pred)):
   if normalize_flag:
       y_pred_denormalized = scalers['temp_to_estimate'].inverse_transform(y_pred[i].reshape(-1, 1)).flatten()
       y_test_denormalized = scalers['temp_to_estimate'].inverse_transform(y_test[i][timesteps-1].reshape(-1, 1)).flatten()
       print(y_pred_denormalized, ",", y_test_denormalized)
    
   else:
       print(y_pred[i].reshape(-1, 1), ",", y_test[i][timesteps-1].reshape(-1, 1))
