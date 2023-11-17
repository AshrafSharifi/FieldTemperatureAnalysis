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
    
    # Define a threshold to detect sharp changes
    # threshold = 2.69

    sensorList = list(range(1,9))
    min_temp = 0
    max_temp = 0
    
    del_items = ['2021_9_1_','2021_9_10_','2021_9_13_','2021_9_22_']
    s1 = []
    s5=[]
    for sensor in sensorList:
        
        
        output_csv_file = "data/sensor_t" + str(sensor) + ".csv"
        # del_csv_file = "data/DB/DB_with_deleted_rows/sensor_t" + str(sensor) + ".csv"
        
        sensor_data = pd.read_csv(output_csv_file)
        # del_sensor_data = pd.read_csv(del_csv_file)
        
        for de in del_items:
            
            matching_rows = sensor_data[sensor_data['complete_timestamp(YYYY_M_DD_HH_M)'].str.startswith(de, na=False)].index
            sensor_data = sensor_data.drop(matching_rows)
            sensor_data.to_csv(output_csv_file, index=False, encoding='utf-8')
            
            # matching_rows = del_sensor_data[del_sensor_data['complete_timestamp(YYYY_M_DD_HH_M)'].str.startswith(de)].index
            # del_sensor_data = del_sensor_data.drop(matching_rows)
            # del_sensor_data.to_csv(del_csv_file, index=False, encoding='utf-8')
        
       
        grouped = sensor_data.groupby(['year','month','day_of_month'])
        # group_sizes = grouped.size().reset_index(name='GroupSize')
        valid_groups = grouped.filter(lambdsa x: len(x) ==96)
        print("s"+ str(sensor) + "_" + str(len(grouped.groups.keys())))
        
        if sensor ==1:
            s1=list(grouped.groups.keys())
        elif sensor == 5:
            s5=list(grouped.groups.keys())
