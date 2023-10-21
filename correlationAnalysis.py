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
    sensor_number = 1  
    
  
    sensor_data = pd.read_csv("data/sensor_t" + str(sensor_number) + ".csv")
    sensorList = [1]
    fin_name = 'data/aggregated_data'
    with open(fin_name, 'rb') as fin:
        aggregated_data = pickle.load(fin)
    fin.close() 
  
    for index, data in aggregated_data[1].items():
        sensor_data = data["sensorData"]
        numerical_data = sensor_data.select_dtypes(include=[np.number])
        diff_data=numerical_data.diff()
        func = functions(diff_data)
        func.corr_matrix(list(diff_data.columns))
        
