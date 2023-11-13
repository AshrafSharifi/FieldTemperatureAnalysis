from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import csv
import pickle

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None 
    sensorList = list(range(1,9))
   
    for sensor in sensorList: 
        print("======================== Sensor: "+ str(sensor) +"=============================")
        fin_name = 'data/aggregated_data' + str(sensor)
        with open(fin_name, 'rb') as fin:
            aggregated_data = pickle.load(fin)
        fin.close() 
    
        # Create a list of data
        data = []
        # Define custom column headers
        headers = []
        for i in range(24):
           headers.append(str(i)) 
        
        
        # Create a DataFrame using pandas
        df = pd.DataFrame(0, index=range(1), columns=headers)
        
    
    
        for index, data in aggregated_data[sensor].items():
            sharpChangeTimes = data["SharpChangeTimes"]
            for i in sharpChangeTimes:
                df[str(i)]=df[str(i)]+1
     
    
        print('size of sample set:', str(len(aggregated_data[sensor])))
        non_zero_values = df.values.ravel()
        non_zero_column_indices = non_zero_values.nonzero()[0]
        non_zero_values = non_zero_values[non_zero_column_indices]
        sorted_indices = non_zero_values.argsort()[::-1]
        sorted_values = non_zero_values[sorted_indices]
        column_names = df.columns[non_zero_column_indices][sorted_indices]
        # Display non-zero values and their corresponding column names
        result_df = pd.DataFrame({'Hour with most changes': column_names, 'Frequency': sorted_values})
        print(result_df)
        
        tempChangesCorr = pd.Series()
        count=0;
        for index, data in aggregated_data[sensor].items():
            if sum(data["corr_temperatureChange"].fillna(0))!=0:
                count += 1
                if index==1:
                    tempChangesCorr = data["corr_temperatureChange"].fillna(0)
                else:
                    tempChangesCorr += data["corr_temperatureChange"].fillna(0)
            
        print('Average of temperature change correlation matrix')
        print(tempChangesCorr/count)
        
   
    
   
