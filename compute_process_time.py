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
from DQN import *


def print_path(path):
    headers = ["POI_number", "Status", "Priority", "Reward", "Temp_difference", "Reach_time", "State","temperature"]
    table = []
    for key, values in path.items():
        if values[0] != 'NotVisited':
            table.append([key, *values])
            
    table = sorted(table, key=lambda x: (x[6][0][3],x[6][0][4], x[6][0][5]))


    print(tabulate(table, headers, tablefmt="pretty"))
    
    
def time_difference(time1, time2):
    # Convert times to minutes
    hours1, minutes1 = map(int, time1.split(':'))
    hours2, minutes2 = map(int, time2.split(':'))
    
    # Calculate the difference in minutes
    total_minutes1 = hours1 * 60 + minutes1
    total_minutes2 = hours2 * 60 + minutes2
    difference_in_minutes = total_minutes2 - total_minutes1
    
    # Convert the difference back to "hour:min" format
    hours_difference = difference_in_minutes // 60
    minutes_difference = difference_in_minutes % 60
    
    # return f"{hours_difference:02d}:{minutes_difference:02d}"

    return difference_in_minutes
def remove_outliers(data):
    # Calculate the first and third quartiles (Q1 and Q3)
    Q1 = sorted(data)[int(len(data) * 0.25)]
    Q3 = sorted(data)[int(len(data) * 0.75)]

    # Calculate the IQR
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers from the list
    cleaned_data = [(index, value) for index, value in enumerate(data) if lower_bound <= value <= upper_bound]
    outliers_data_with_indices = [(index, value) for index, value in enumerate(data) if lower_bound > value or value > upper_bound]


    return cleaned_data,outliers_data_with_indices



process_time=True
with open('data/path_data', 'rb') as fin:
    path_data = pickle.load(fin)
fin.close()

if process_time:
    
    t_list=['current']

    for t_item in t_list:
        if(t_item=='current'):
            with open('data/all_traj', 'rb') as fin:
                all_traj = pickle.load(fin)
                day = all_traj[0][0][3]
        else:
            with open('data/prev_all_traj', 'rb') as fin:
                all_traj = pickle.load(fin)
                day = all_traj[0][3]
                
        
        counter = 0
        total_time_path = []
        total_time_rew = []
        
        if t_item=='current':
            for i in range(2, len(all_traj)-1, 2):
                current_element =np.reshape(all_traj[i], [1, 6]) 
                next_element = all_traj[i + 1]
                counter+=2

                    
                current_element=np.reshape(current_element, [1, 6])  
                next_element=np.reshape(next_element, [1, 6])  
    
                try:
                    # Create datetime instances
                    datetime1 = datetime(current_element[0,1], current_element[0,2], current_element[0,3], current_element[0,4], current_element[0,5])
                    datetime2 = datetime(next_element[0,1], next_element[0,2], next_element[0,3], next_element[0,4], next_element[0,5])
                    
                    # Calculate time difference
                    time_difference = datetime2 - datetime1
                    
                    # Extract days, hours, and minutes from the time difference
                    result = (datetime2 - datetime1).total_seconds() / 60

                    total_time_path.append(result)
                    path_temp = path_data[counter+1]
                    total_time_rew.append(sum(row[2] for row in path_temp.values()))
                    path_temp = path_data[counter+1]
                    

                    
                    # if result==105:
                    #     path_temp = path_data[counter+1]
                    #     print('---min time')
                    #     print_path(path_temp)
                    # if result==150:
                    #     path_temp = path_data[counter+1]
                    #     print('---max time')
                    #     print_path(path_temp)
                except:
                    continue


        
        if(t_item=='current'):
            with open('data/total_time_path', 'wb') as fout:
                pickle.dump([x for x in total_time_path if x >= 0], fout)  
        else:
            with open('data/prev_total_time_path', 'wb') as fout:
                pickle.dump([x for x in total_time_path if x >= 0], fout)  
        
    

        
        
else:
    
    with open('data/total_time_path', 'rb') as fin:
        total_time_path = pickle.load(fin)
    fin.close() 
    with open('data/path_data', 'rb') as fin:
        path_data = pickle.load(fin)
    fin.close() 
    cleaned,outliers =remove_outliers(total_time_path)

    
    total_time_path = [item[1] for item in cleaned if item[1]>75]
    print(len(total_time_path))
    average_time_per_path = sum(total_time_path)/len(total_time_path)
    total_time_path = sorted(total_time_path)
    print('average:' + str(average_time_per_path))
    print('min:' + str(min(total_time_path)))
    print('max:' + str(max(total_time_path)))
    # index = total_time_path.index(min(total_time_path))
    # path=path_data[index-1]
    # print_path(path)
             
    
    with open('data/prev_total_time_path', 'rb') as fin:
        prev_total_time_path = pickle.load(fin)
    fin.close() 
   
    process_time_total= len(prev_total_time_path)*7*15
    average_time_per_prev_path = (sum(prev_total_time_path)+process_time_total) /len(prev_total_time_path)
          
            
    
    print(average_time_per_prev_path)

        




    
    
    
 