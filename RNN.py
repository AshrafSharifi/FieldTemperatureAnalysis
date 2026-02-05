import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ================== CONFIG ==================

train = True  # True: train model, False: test model
normalize_flag = False
print_flag = False

DB_train = "data/DB/train_DB.csv"
DB_test = "data/DB/test_DB.csv"
model_file = "data/RNN_models/temperature_prediction_model.hdf5"

data_header = [
    'sensor',
    'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
    'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
]

epochs = 400
batch_size = 32
validation_split = 0.2
timesteps = 3
patience = 100
dropout = 0.2
learning_rate = 0.0001

ts_col = 'complete_timestamp(YYYY_M_DD_HH_M)'
target_col = 'temp_to_estimate'


# ================== UTILS ==================

def sort_key(entry):
    timestamp_str = entry[3]
    timestamp_obj = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M')
    return timestamp_obj


def change_lable(D):
    year = int(D[0])
    month = int(D[1])
    day = int(D[2])
    hours = int(D[3])
    minutes = int(D[4])

    minute_mapping = {0: 0, 1: 15, 2: 30, 3: 45}
    mapped_minutes = minute_mapping[minutes]
    result = f"{year:04d}/{month:02d}/{day:02d}_{hours:02d}:{mapped_minutes:02d}"
    return result


def plot_result_for_each_POI(actual_values, prediction_values, POI, result):
    sample_num = min(10, len(result))
    result_items = [change_lable(item[3].split('_')) for item in result][:sample_num]
    actual_values = [item for item in actual_values[:sample_num]]
    prediction_values = [item for item in prediction_values[:sample_num]]

    bar_width = 0.25
    index = np.arange(sample_num)

    fig, ax = plt.subplots()
    ax.bar(index, actual_values, bar_width, label='Actual Values', color='dodgerblue')
    ax.bar(index + bar_width, prediction_values, bar_width, label='Predicted Values', color='deeppink')

    for i, v in enumerate(actual_values):
        ax.text(i, v + 0.6, str(round(v, 1)), color='blue', ha='center',
                rotation=90, va='bottom', fontsize=8, weight='bold')
    for i, v in enumerate(prediction_values):
        ax.text(i + bar_width, v + 0.6, str(round(v, 1)), color='m', ha='center',
                rotation=90, va='bottom', fontsize=8, weight='bold')

    ax.set_ylabel('Temperature °C')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(result_items, rotation=45, ha='right')
    ax.legend()
    plt.ylim(0, 35)
    plt.show()


def compute_metrics(predicted_values, actual_values):
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    loss = np.sum((predicted_values - actual_values) ** 2)
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")


def analyze_result_per_POIs(predicted_values, actual_values, sensors):
    print('---Results For All---')
    compute_metrics(predicted_values, actual_values)

    sensors = pd.Series(sensors)
    for POI in range(1, 8):
        idx = sensors[sensors == POI].index
        if len(idx) == 0:
            continue
        pred = predicted_values[idx]
        act = actual_values[idx]
        print(f'---Results For POI {POI}---')
        compute_metrics(pred, act)


def analyze_result_per_POIs_old(resul):
    predicted_values = [item[1][0] for item in resul]
    actual_values = [item[2] for item in resul]
    print('---Results For All (old)---')
    compute_metrics(np.array(predicted_values), np.array(actual_values))

    for POI in range(1, 8):
        result_items = [item for item in resul if item[0] == POI]
        result_items = sorted(result_items, key=sort_key)
        if len(result_items) == 0:
            continue
        predicted_values = [item[1][0] for item in result_items]
        actual_values = [item[2] for item in result_items]
        print('---Results For POI ', str(POI))
        compute_metrics(np.array(predicted_values), np.array(actual_values))
        plot_result_for_each_POI(actual_values, predicted_values, POI, result_items)


def create_sequences(dataset, seq_length):
    """
    dataset: (N, n_features), last column = target
    seq_length: number of time steps
    """
    X, Y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i + seq_length, :-1])   # all features except target
        Y.append(dataset[i + seq_length, -1])      # next-step target
    return np.array(X), np.array(Y)


def create_model(timesteps, n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True,
                                 input_shape=(timesteps, n_features))))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='mean_absolute_percentage_error',
                  metrics=['mape', 'mae', 'mse'])
    return model


def load_and_prepare(DB_path, normalize_flag):
    df = pd.read_csv(DB_path)
    df = df[data_header].dropna()

    df[ts_col] = pd.to_datetime(df[ts_col], format='%Y_%m_%d_%H_%M')
    df.sort_values(ts_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if normalize_flag:
        scalers = {}
        for col in ['temp_to_estimate', 'temp_centr']:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
    else:
        scalers = None

    df_numeric = df.drop(columns=[ts_col])
    # ensure target is last column
    cols = [c for c in df_numeric.columns if c != target_col] + [target_col]
    df_numeric = df_numeric[cols]

    data = df_numeric.values.astype(np.float32)
    return df, df_numeric, data, scalers


# ================== MAIN ==================

if train:
    df_train, df_num_train, data_train, scalers_train = load_and_prepare(DB_train, normalize_flag)

    x, y = create_sequences(data_train, timesteps)
    # chronological split (Option C style)
    split_idx = int(len(x) * 0.8)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    n_features = x_train.shape[2]
    model = create_model(timesteps, n_features)

    checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[checkpointer, early_stopping],
        verbose=1
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Val MSE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

    plt.figure()
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.show()

    plt.figure()
    plt.plot(history.history['mape'], label='Train MAPE')
    plt.plot(history.history['val_mape'], label='Val MAPE')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.show()

    test_results = model.evaluate(x_val, y_val, verbose=0)
    print('-------------- Validation results -----------------')
    print(test_results)

else:
    model = load_model(model_file)

    df_test, df_num_test, data_test, scalers_test = load_and_prepare(DB_test, normalize_flag)
    x_test, y_test = create_sequences(data_test, timesteps)

    preds = model.predict(x_test, verbose=0).flatten()
    actual = y_test.flatten()

    # align sensors with sequence targets (each y at index i+timesteps)
    sensor_series = df_num_test['sensor'].values
    sensors_for_samples = sensor_series[timesteps:timesteps + len(actual)]

    compute_metrics(preds, actual)
    analyze_result_per_POIs(preds, actual, sensors_for_samples)

    # optional: build "result" structure for old POI function
    timestamps = df_test[ts_col].dt.strftime('%Y_%m_%d_%H_%M').values
    ts_for_samples = timestamps[timesteps:timesteps + len(actual)]
    result = [[int(sensors_for_samples[i]), np.array([preds[i]]), actual[i], ts_for_samples[i]]
              for i in range(len(actual))]
    # you can still call:
    # analyze_result_per_POIs_old(result)
