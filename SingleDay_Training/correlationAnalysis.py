from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import pickle

if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None 
    display = False
    sensorList = list(range(1,9))
    
    for sensor in [sensorList]:     
        sensor_data = pd.read_csv("data/sensor_t" + str(sensor) + ".csv")
        fin_name = 'data/aggregated_data'+ str(sensor)
        with open(fin_name, 'rb') as fin:
            aggregated_data = pickle.load(fin)
        fin.close() 
      
        for index, data in aggregated_data[sensor].items():
            sensor_data = data["sensorData"]
            numerical_data = sensor_data.select_dtypes(include=[np.number])
            diff_data=numerical_data.diff()
            func = functions(diff_data)
            corr_matrix = func.corr_matrix(list(diff_data.columns))
            data["corr"]=corr_matrix;
            data["corr_temperatureChange"]=corr_matrix["temperatureChange"];
            
            if (display==True):
                _, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
                # print(corr_matrix)
                sns.heatmap(corr_matrix,ax=axes);
        with open('data/aggregated_data'+ str(sensor), 'wb') as fout:
            pickle.dump(aggregated_data, fout)
        fout.close()
        
