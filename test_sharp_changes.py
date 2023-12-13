from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import csv
import pickle

def aggregate_sharp_changes():
    
    fin_name = 'data/aggregated_data'
    with open(fin_name, 'rb') as fin:
        aggregated_data = pickle.load(fin)
    fin.close() 
    
    count = 1
    sharpChangesTimes = dict()
    for index, data in aggregated_data[1].items():
        sharpChangesTimes[count] = data["SharpChangeTimes"];
        count += 1

    
        
    
def map_time_stamp(complete_timestamp):
    # Split the timestamp based on '_'
    timestamp_parts = complete_timestamp.split('_')
    
    # Extract the last two values
    hour = int(timestamp_parts[-2])
    minute = int(timestamp_parts[-1])
    
    # Map the last character to the corresponding minute value
    minute_mapping = {
        '0': '00',
        '1': '15',
        '2': '30',
        '3': '45'
    }
    
    # Map the last character to the corresponding minute value, or default to '00'
    minute_str = minute_mapping.get(str(minute), '00')
    result = f"{hour}:{minute_str}"
    return result

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None 
    
    # Define a threshold to detect sharp changes
    # threshold = 2.69

    sensorList = list(range(1,9))
    min_temp = 0
    max_temp = 0
    plt.figure()
    for sensor in [3,7]:
        sensor_data = pd.read_csv("data/sensor_t" + str(sensor) + ".csv")
        

        # -------remove data of days which has small size
        grouped = sensor_data.groupby(['year','month','day_of_month'])
        group_sizes = grouped.size().reset_index(name='GroupSize')
        valid_groups = grouped.filter(lambda x: len(x) == 96)
        indices_to_keep = valid_groups.index
        sensor_data = sensor_data.loc[indices_to_keep]
        # ----------------------------------------------
        
        
        func = functions(sensor_data,0, False, False)
        sensor_data = func.change_outlier(sensor_data)
        
        if min(sensor_data["temp_to_estimate"]) < min_temp:
            min_temp = min(sensor_data["temp_to_estimate"])
        
        if max(sensor_data["temp_to_estimate"]) > max_temp:
            max_temp = max(sensor_data["temp_to_estimate"])
            
        threshold = func.find_threshold_based_on_variation()
        
        
        sensorData = dict()
        allData = dict()
        aggregated_data = dict()
        
        count = 1 
        sensorData = dict()
        for year in sensor_data['year'].unique():
            # Filter data based on the specified date
            idx = (sensor_data['year'] == year)
            filtered_data_step1 = sensor_data.loc[idx,:]
            for month in [filtered_data_step1['month'].unique()[1]]:
                idx = (filtered_data_step1['month'] == month) 
                filtered_data_step2 = filtered_data_step1.loc[idx,:]
                for day in [filtered_data_step2['day_of_month'].unique()[6]]:
                    idx = (filtered_data_step2['day_of_month'] == day)
                    filtered_data = filtered_data_step2.loc[idx,:]
                    
                    filtered_data = filtered_data[(filtered_data['hour'] >= 14) & (filtered_data['hour'] <= 15)]
                    
                    time_vector=filtered_data['hour'].values
                    
                    time_vector = list(map(map_time_stamp, filtered_data['complete_timestamp(YYYY_M_DD_HH_M)']))
                    temp_values= list(filtered_data['temp_to_estimate'])
                    
                    if sensor==3:
                        plt.plot(time_vector, temp_values, 'b', linewidth=2)
                    else:
                        plt.plot(time_vector, temp_values, 'r', linewidth=2)
                        
                    # plt.scatter(time_vector[sharp_change_indices], temperature_data[sharp_change_indices], c='r', marker='o')
                    plt.xlabel('Time Intervals')
                    plt.ylabel('Temperature °C')
     
                    count += 1


    plt.grid(True)
    plt.ylim(20, 35)
    # plt.xticks(fontsize=8)    
    plt.show()
    temp_info = dict()
    temp_info['min_temp'] = min_temp
    temp_info['max_temp'] = max_temp
    
    # with open('data/temp_info', 'wb') as fout:
    #     pickle.dump(temp_info, fout)
    # fout.close()
    
    # aggregate_sharp_changes();
    

