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

    
        
    


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None 
    
    # Define a threshold to detect sharp changes
    # threshold = 2.69

    sensorList = list(range(1,9))
    min_temp = 0
    max_temp = 0
    for sensor in sensorList:
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
            for month in filtered_data_step1['month'].unique():
                idx = (filtered_data_step1['month'] == month) 
                filtered_data_step2 = filtered_data_step1.loc[idx,:]
                for day in filtered_data_step2['day_of_month'].unique():
                    idx = (filtered_data_step2['day_of_month'] == day)
                    filtered_data = filtered_data_step2.loc[idx,:]
                    func = functions( filtered_data, threshold=threshold,sensor=0, displayResult=True, hasOutput=True)
                    data = func.data_preperation()
                    output = func.first_derivative()
                    allData = dict()
                    allData["sensor"] = sensor;
                    allData["sensorData"] = filtered_data;
                    allData["preproccessedData"] = data;
                    allData["year"] = year;
                    allData["month"] = month;
                    allData["day"] = day;
                    # allData["sharpChangedData"] = output;
                    if output == None:
                        allData["SharpChangeValues"] = [] 
                        allData["SharpChangeIndices"] = []
                        allData["SharpChangeTimes"] = []
                    else:
                        allData["SharpChangeValues"] = output["SharpChangeValues"] 
                        allData["SharpChangeIndices"] = output["SharpChangeIndices"]
                        allData["SharpChangeTimes"] = output["SharpChangeTimes"] 
                    sensorData[count] = allData;
                    count += 1

        aggregated_data[sensor]= sensorData
        # Save the dictionary to a CSV file
    # with open("aggregated_data.csv", "w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in aggregated_data.items():
    #         writer.writerow([key, value])
        with open('data/aggregated_data'+ str(sensor), 'wb') as fout:
            pickle.dump(aggregated_data, fout)
        fout.close()
        
    
    temp_info = dict()
    temp_info['min_temp'] = min_temp
    temp_info['max_temp'] = max_temp
    
    with open('data/temp_info', 'wb') as fout:
        pickle.dump(temp_info, fout)
    fout.close()
    
    # aggregate_sharp_changes();
    

