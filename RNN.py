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
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
import random

# Define the ranges for hyperparameters
learning_rate_range = [0.0001, 0.001, 0.01]
batch_size_range = [16, 32, 64]
epochs_range = [200, 300, 400]
patience_range =[50,100,150]
dropout_range=[0.2,0.55,0.3]
timesteps_range = [3,4,5]
current_DB = True
train = False
normalize_flag = False
print_flag = False
validation_split = 0.2
best_result = float('inf')
best_iteration = 0
best_model = None
best_x_train = None
best_x_test = None
best_y_train = None
best_y_test = None
best_scalers = None   
train_for_specific_config = False 
with_out_process_time=False
# Data preprocessing
data_header = [
    'sensor', 
    'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
    'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
]

def localize_row(df_base,state,time_stamp):
    
    
    
    
    row = df_base[df_base['complete_timestamp(YYYY_M_DD_HH_M)']== 	time_stamp]
    
    new_column_header = 'sensor'
    new_column_value = state[0]
    
    # Add the new column to the DataFrame
    row = row.assign(**{new_column_header: new_column_value})

    row= row[data_header]
    
    return row
    
from datetime import datetime
def sort_key(entry):
    timestamp_str = entry[3]  # Assuming timestamp is always at the 4th position
    timestamp_obj = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M')
    return timestamp_obj


def analyze_result_per_POIs(resul):
    
    predicted_values = [item[1][0] for item in resul]
    actual_values = [item[2] for item in resul]
    print('---Results For All ')
    compute_metrics(np.array(predicted_values), np.array(actual_values))
    
    
    for POI in range(1,8):
        result_items = [item for item in result if item[0] == POI]
        result_items = sorted(result_items, key=sort_key)
        predicted_values = [item[1][0] for item in result_items]
        actual_values = [item[2] for item in result_items]
        print('---Results For POI ',str(POI))
        compute_metrics(np.array(predicted_values), np.array(actual_values))
        plot_result_for_each_POI(actual_values,predicted_values,POI,result_items)

def change_lable(D):
    year=int(D[0])
    month=int(D[1])
    day=int(D[2])
    hours = int(D[3])
    minutes = int(D[4])
    
    # Map minutes to quarters
    minute_mapping = {
        0: 0,
        1: 15,
        2: 30,
        3: 45
    }
    
    # Get the corresponding minute value
    mapped_minutes = minute_mapping[minutes]
    
    # Format the result
    result = f"{year:04d}/{month:02d}/{day:02d}_{hours:02d}:{mapped_minutes:02d}"
    
    return result
            
def plot_result_for_each_POI(actual_values,prediction_values,POI,result):
    
    # Sample data
    sample_num = 10
    result_items = [change_lable(item[3].split('_')) for item in result][:sample_num]
    actual_values = [item[0] for item in actual_values[:sample_num]]  # Extract values from the arrays
    prediction_values = [item[0] for item in prediction_values[:sample_num]]
    
    # Calculate the differences between actual and predicted values
    differences = np.array(actual_values) - np.array(prediction_values)
    
   # Plotting as a bar chart with lines connecting actual and predicted values
    bar_width = 0.25
    index = np.arange(sample_num)
    
    fig, ax = plt.subplots()
    
    # Plot bars and set colors
    bar1 = ax.bar(index, actual_values, bar_width, label='Actual Values', linestyle='-', color='dodgerblue')
    bar2 = ax.bar(index + bar_width, prediction_values, bar_width, label='Predicted Values', color='deeppink')
    
    # Color specific bars differently
    # bar1[0].set_color('darkpink')  # Change the color of the first bar to dark pink
    
       # Add labels on top of the bars
    for i, v in enumerate(actual_values):
        ax.text(i, v + 0.6, str(round(v, 1)), color='blue', ha='center',rotation=90, va='bottom',  fontsize=8,weight='bold')
    
    for i, v in enumerate(prediction_values):
        ax.text(i + bar_width, v + 0.6, str(round(v, 1)), color='m', ha='center',rotation=90, va='bottom',  fontsize=8,weight='bold')

    
    # Other plot configurations
    ax.set_ylabel('Temperature °C')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(result_items, rotation=45, ha='right')
    ax.legend()
    plt.ylim(0, 35)
    plt.show()
    S=1





    
        
def plot_result_for_each_month(actual_values,prediction_values,df):
    
     
    df=pd.DataFrame(df)
    unique_months = df[2].unique()
    for month in unique_months:
        plt.figure(figsize=(12, 6))
        
        # Filter data for the specific month
     
        month_data = df[df.iloc[:, 2] == month]
        month_indices = month_data.index.values
        
        # Index the actual and prediction values using the month indices
        month_actual_values = actual_values[:, month_indices]
        month_prediction_values = prediction_values[:, month_indices]
    
        plt.scatter(range(len(month_actual_values.flatten())), month_actual_values.flatten(), label='Actual Values', marker='o', alpha=0.7)
        plt.scatter(range(len(month_prediction_values.flatten())), month_prediction_values.flatten(), label='Predicted Values', marker='x', alpha=0.7)
        plt.plot(month_actual_values.flatten(), label='_nolegend_', linestyle='-', color='blue', alpha=0.5)
        plt.plot(month_prediction_values.flatten(), label='_nolegend_', linestyle='-', color='orange', alpha=0.5)
        plt.xlabel('Data Points')
        plt.ylabel('Temperature')
        plt.title(f'Actual vs Predicted Values - Month {int(month)}')
        plt.legend()
        plt.show()
    
    
def compute_metrics(predicted_values,actual_values):
    

    
    # Calculate loss, MSE, MAE, and MAPE
    loss = np.sum((predicted_values - actual_values) ** 2)
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Print the results
    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    
def create_sequences(dataset, seq_length):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(dataset[i][:seq_length - 1])
        y.append(dataset[i][seq_length - 1:])
    return np.array(x), np.array(y)

def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
    return model
    
    



if current_DB:
    DB_file = "data/current_work_with_process_time/DB/train_DB.csv"
    if with_out_process_time:
        if normalize_flag:
            hyper_params_name = 'data/current_work_without_process_time/RNN_models/withNormalization/hyper_parameters'
            model_file = 'data/current_work_without_process_time/RNN_models/withNormalization/temperature_prediction_model.hdf5'
            best_result_path = 'data/current_work_without_process_time/RNN_models/withNormalization/best_trained_data'
        else:
            hyper_params_name = 'data/current_work_without_process_time/RNN_models/without_Normalization/hyper_parameters'
            model_file = 'data/current_work_without_process_time/RNN_models/without_Normalization/temperature_prediction_model.hdf5'
            best_result_path = 'data/current_work_without_process_time/RNN_models/without_Normalization/best_trained_data'   
    else:    
        if normalize_flag:
            hyper_params_name = 'data/current_work_with_process_time/RNN_models/withNormalization/hyper_parameters'
            model_file = 'data/current_work_with_process_time/RNN_models/withNormalization/temperature_prediction_model.hdf5'
            best_result_path = 'data/current_work_with_process_time/RNN_models/withNormalization/best_trained_data'
        else:
            hyper_params_name = 'data/current_work_with_process_time/RNN_models/without_Normalization/hyper_parameters'
            model_file = 'data/current_work_with_process_time/RNN_models/without_Normalization/temperature_prediction_model.hdf5'
            best_result_path = 'data/current_work_with_process_time/RNN_models/without_Normalization/best_trained_data'
else:
    DB_file = "data/previous_work/DB/train_DB.csv"
    if normalize_flag:
        hyper_params_name = 'data/previous_work/RNN_models/withNormalization/hyper_parameters'
        model_file = 'data/previous_work/RNN_models/withNormalization/temperature_prediction_model.hdf5'
        best_result_path = 'data/previous_work/RNN_models/withNormalization/best_trained_data'
    else:
        hyper_params_name = 'data/previous_work/RNN_models/without_Normalization/hyper_parameters'
        model_file = 'data/previous_work_with/RNN_models/without_Normalization/temperature_prediction_model.hdf5'
        best_result_path = 'data/previous_work/RNN_models/without_Normalization/best_trained_data'
        


if train_for_specific_config:
   
    # with open(hyper_params_name, 'rb') as fin:
    #     hyper_params = pickle.load(fin)
    # fin.close()
    # config = (min(hyper_params.items(), key=lambda x: x[1]['result'][3]))[1]
    # # sorted_configs = sorted(hyper_params.values(), key=lambda x: x['result'][3])
    # # config=sorted_configs[0]    
    # learning_rate =  config['learning_rate']
    # batch_size = config['batch_size']
    # epochs = config['epochs']
    # patience=config['patience']
    # dropout=config['dropout']
    # timesteps =config['timesteps']
    # normalize_flag=config['normalize_flag']
    # validation_split= config['validation_split']
    # epochs=150


    
    normalize_flag= False
    epochs= 400
    batch_size= 32
    validation_split= 0.2
    timesteps= 3
    patience= 100
    dropout= 0.2
    learning_rate= 0.0001


        
        
        
    # Load the data
    file = DB_file
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
    # model.summary()
    checkpointer = ModelCheckpoint(filepath = model_file, verbose = 2, save_best_only = True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    # Train the model       
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks = [checkpointer,early_stopping])
    
    



    # Visualize the training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Visualize the training loss
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    # Visualize the training loss
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['mape'], label='Training MAPE')
    plt.plot(history.history['val_mape'], label='Validation MAPE')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()
    
    # Evaluate the model on the test set
    test_results = model.evaluate(x_test, y_test)
    print('--------------Test results:-----------------')
    print(test_results)
    
    
    if current_DB:
        root = 'data/Test_CurrentMethod/'
    else:
        root = 'data/Test_PrevMethod/'
                
                
    with open(root+'x_test', 'wb') as fout:
        pickle.dump(x_test, fout)
    fout.close()
    
    with open(root+'y_test', 'wb') as fout:
        pickle.dump(y_test, fout)
    fout.close()

    
    # # Make predictions and denormalize
    # y_pred = model.predict(x_test)
    # print("Prediction, real value")
    # initial_x_test=np.array([]).reshape(0, x_test.shape[2]) 
    # prediction_values=[]
    # actual_values=[]
    # result=list()
    # for i in range(len(y_pred)):
    #    temp=(x_test[i][timesteps-1].reshape(-1, 1)).T
    #    initial_x_test = np.vstack([initial_x_test, np.array(temp[0])])
  
    #    if normalize_flag:
    #        y_pred_denormalized = scalers['temp_to_estimate'].inverse_transform(y_pred[i].reshape(-1, 1)).flatten()
    #        y_test_denormalized = scalers['temp_to_estimate'].inverse_transform(y_test[i][timesteps-1].reshape(-1, 1)).flatten()
           
    #        prediction_values.append(y_pred_denormalized[0])
    #        actual_values.append(y_test_denormalized[0][0])
           
    #        print(y_pred_denormalized[0], ",", y_test_denormalized[0][0])
        
    #    else:
    #        predicted_value=y_pred[i].flatten()[0]
    #        real_value=y_test[i][timesteps-1].reshape(-1, 1)[0][0]
    #        prediction_values.append(predicted_value)
    #        actual_values.append(real_value)
           
    #        print(predicted_value, ",", real_value)
    #        result.append([int(initial_x_test[i,0]),predicted_value,real_value])
    
    # actual_values = np.reshape(actual_values, (1,len(actual_values)))
    # prediction_values = np.reshape(prediction_values, (1,len(prediction_values)))
    # compute_metrics(prediction_values,actual_values)
    # plot_result_for_each_month(actual_values,prediction_values,initial_x_test)
    # analyze_result_per_POIs(result)  
    
    
else:        
    for iteration in range(1,40):
        print("----------------"+str(iteration)+"-----------------")
        
        
    
        # Generate random hyperparameter values
        learning_rate =  random.choice(learning_rate_range)
        batch_size = random.choice(batch_size_range)
        epochs = random.choice(epochs_range)
        patience=random.choice(patience_range)
        dropout=random.choice(dropout_range)
        timesteps =random.choice(timesteps_range)
        
        
    
    
        try:
            with open(hyper_params_name, 'rb') as fin:
                hyper_parameters = pickle.load(fin)
            fin.close() 
        except:
            hyper_parameters = dict()
        
        
        
        hyper_params = dict()
        hyper_params['normalize_flag'] = normalize_flag
        hyper_params['epochs'] = epochs
        hyper_params['batch_size'] = batch_size
        hyper_params['validation_split'] = validation_split
        hyper_params['timesteps'] = timesteps
        hyper_params['patience'] = patience
        hyper_params['dropout'] = dropout
        hyper_params['learning_rate'] = learning_rate
        
    
     
        
        
       
        
        
        if train:
            # Data preprocessing
            data_header = [
                'sensor', 
                'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
                'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
            ]
            
            
            
            # Load the data
            file = DB_file
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
            # model.summary()
            checkpointer = ModelCheckpoint(filepath = model_file, verbose = 2, save_best_only = True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            # Train the model       
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks = [checkpointer,early_stopping])
            
            
            # Save the trained model
            if history.history['val_loss'][-1] < best_result:
                best_result = history.history['val_loss'][-1]
                best_iteration = iteration
                best_model = model
                best_x_train = x_train
                best_x_test = x_test
                best_y_train = y_train
                best_y_test = y_test
                best_scalers = scalers
    
    
    
            # Save the best model, training, and test sets
            best_trained_data = {
                'x_train': best_x_train,
                'x_test': best_x_test,
                'y_train': best_y_train,
                'y_test': best_y_test,
                'scalers': best_scalers
            }
            
            with open(best_result_path, 'wb') as fout:
                pickle.dump(best_trained_data, fout)
            fout.close()
            
            best_model.save(model_file)
    
    
            # Visualize the training loss
            # plt.plot(history.history['loss'], label='Training Loss')
            # plt.plot(history.history['val_loss'], label='Validation Loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show()
            
            # Evaluate the model on the test set
            test_results = model.evaluate(x_test, y_test)
            print('--------------Test results:-----------------')
            print(test_results)
            
    
            hyper_params["result"] = test_results
            hyper_parameters[str(len(hyper_parameters)+1)] = hyper_params
            with open(hyper_params_name, 'wb') as fout:
                pickle.dump(hyper_parameters, fout)
            fout.close()
            
            
        else:
        # Load the trained model for testing
        
            do_test=False
            if current_DB:
                root = 'data/Test_CurrentMethod/'
            else:
                root = 'data/Test_CurrentMethod/'
            
            if do_test:
                model = load_model(model_file)
        
                # with open(best_result_path, 'rb') as fin:
                #     trained_data = pickle.load(fin)
                # fin.close()
                
                # with open(hyper_params_name, 'rb') as fin:
                #     hyper_params = pickle.load(fin)
                # fin.close()
                result=list()
                timesteps=3
                # config = (min(hyper_parameters.items(), key=lambda x: x[1]['result'][3]))[1]
                # config1 = (min(hyper_parameters.items(), key=lambda x: x[1]['result'][0]))[1]
                # timesteps = config['timesteps']
                # x_train = trained_data['x_train']
                # x_test = trained_data['x_test']
                # y_train = trained_data['y_train']
                # y_test = trained_data['y_test']
                # scalers = trained_data['scalers']
            
                # Evaluate the model on the test set
                
                with open(root+'x_test', 'rb') as fin:
                    x_test = pickle.load(fin)
                fin.close()
                
                with open(root+'y_test', 'rb') as fin:
                    y_test = pickle.load(fin)
                fin.close()
                
                
                test_results = model.evaluate(x_test, y_test)
                print('--------------Test results:-----------------')
                print(test_results)
                
                

    
                
                if True:
                    df = dict()
                    for sensor in range(1,8):
                        csv_file = "data/sensor_t"+str(sensor)+".csv"   
                        df_sensor = pd.read_csv(csv_file)
                        df[sensor]=df_sensor.copy()
                        
                        
    
                    # Make predictions and denormalize
                    all_points = list(range(1,8))
                    prediction_values=[]
                    actual_values=[]
                    sensor_frequently = [0] * 7
                    
                    
                    
                    
    
        
                    counter=0
                    array_test_secondry=[]
                    for i in range(len(y_test)):
                        
                        print(i)
                        x_test_temp = x_test[i].copy()
                        x_test_temp=((x_test_temp[timesteps-1].reshape(-1, 1)).T)[0]
                        temp_points = all_points.copy()
                        temp_points.remove(int(x_test_temp[0]))
                        
                        y= int(x_test_temp[1])
                        m= int(x_test_temp[2])
                        d= int(x_test_temp[5])
                        h= int(x_test_temp[7])
                        t= int(x_test_temp[8])
                        array_test_secondry=[]
                        time_stamp=str(y)+'_'+str(m)+'_'+str(d)+'_'+str(h)+'_'+str(t)
                        for point in temp_points.copy():
                            state_to_find = [point,y,m,d,h,t]
                            row = localize_row(df[point], state_to_find,time_stamp)
                            if counter==0:
                                df_test_secondry= pd.DataFrame(columns=row.columns)
    
                            array_test_secondry.append(np.array(row)[0])
                            
                        df_test_secondry = pd.DataFrame(array_test_secondry,columns=row.columns) 
                        df_test_secondry['complete_timestamp(YYYY_M_DD_HH_M)'] = t
                        
                        data = df_test_secondry[data_header].values
                  
                        
                        data = np.asarray(data).astype(np.float32)
                        x, y = create_sequences(data, len(data_header))
                        
                        for j in range(len(y)): 
                            sample_reshaped = np.tile(x[j], (1, 3, 1))
                            y_pred = model.predict(sample_reshaped)
                            #print(y_pred, ",", y[i])
                            result.append([int(x[j][0]),y_pred, y[j],time_stamp])
                            prediction_values.append(y_pred)
                            actual_values.append(y[j])
                        
                                      
                
    
                         
                         
                with open(root+'result', 'wb') as fout:
                    pickle.dump(result, fout)
                fout.close()
                
                with open(root+'x_test', 'wb') as fout:
                    pickle.dump(x_test, fout)
                fout.close()
                
                with open(root+'y_test', 'wb') as fout:
                    pickle.dump(y_test, fout)
                fout.close()
                analyze_result_per_POIs(result)
            else:
                with open(root+'result', 'rb') as fin:
                    result = pickle.load(fin)
                fin.close()
    
                # actual_values = np.reshape(actual_values, (1,len(actual_values)))
                # prediction_values = np.reshape(prediction_values, (1,len(prediction_values)))
                # compute_metrics(prediction_values,actual_values)
                # plot_result_for_each_POI(actual_values,prediction_values,result)
                analyze_result_per_POIs(result)
                
                       
            break
