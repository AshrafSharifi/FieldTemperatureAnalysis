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

process_time=False

if process_time:
    
    t_list=['current','prev']
    
    for t_item in t_list:
        if(t_item=='current'):
            with open('data/all_traj', 'rb') as fin:
                all_traj = pickle.load(fin)
        else:
            with open('data/prev_all_traj', 'rb') as fin:
                all_traj = pickle.load(fin)
                
        total_time_day = dict()
        day = all_traj[0][3]
        counter = 0
        total_time_path = []
        
        for i in range(0, len(all_traj)-1, 2):
            current_element = all_traj[i]
            next_element = all_traj[i + 1]
            
            if next_element[3] != day:
                total_time_day[str(day)] = total_time_path
                total_time_path = []
                counter = 0
        
            if counter == 0:
                day = next_element[3]
                counter += 1
                continue
        
            time1 = str(current_element[4]) + ':' + str(current_element[5])
            time2 = str(next_element[4]) + ':' + str(next_element[5])
            result = time_difference(time1, time2)
            total_time_path.append(result)
        
        # Store the last day's data
        total_time_day[str(day)] = total_time_path
        
        
        if(t_item=='current'):
            with open('data/total_time_day', 'wb') as fout:
                pickle.dump(total_time_day, fout)  
        else:
            with open('data/prev_total_time_day', 'wb') as fout:
                pickle.dump(total_time_day, fout)  
        
    

        
        
else:
    with open('data/total_time_day', 'rb') as fin:
        total_time_day = pickle.load(fin)
    fin.close() 
    
    days=0
    sum_time=[]
    for item in total_time_day.values():
        if len(item)!=0:
            sum_time.append(sum(item))
            days+=1
    
    average_time=sum(sum_time)/days
    print(average_time/60)
             
    
    with open('data/prev_total_time_day', 'rb') as fin:
        total_time_day = pickle.load(fin)
    fin.close() 
    with open('data/prev_process_in_day', 'rb') as fin:
        prev_process_in_day = pickle.load(fin)
    fin.close() 
    days=0
    prev_sum_time=[]
    for item in total_time_day.values():
        if len(item)!=0:
            prev_sum_time.append(sum(item)+(prev_process_in_day[days]*15))
            days+=1
    
          
            
    prev_average_time=sum(prev_sum_time)/days
    print(prev_average_time/60)

        
 
    
    
    
    # path = "data/previous_work/DB/train_DB.csv"
    # df_temp = pd.read_csv(path)  
       
    
    
    # output=dict()
    # dqn = DQN()
    # for path in DB_file:
    #     # Load the data
        
       
    #     df_temp = pd.read_csv(path)
    #     grouped = df_temp.groupby(['year','month','day_of_month'])
    #     total_process_time=0
    #     total_transition_time= 0
    #     total_total_time=0
    #     for item in grouped.groups.keys():
    #         if str(item) != '(nan, nan, nan)':
                
    #             process_time=0
    #             transition_time= 0
    #             total_time=0
    #             criteria = (df_temp['year'] ==int(item[0])) & (df_temp['month'] ==int(item[1])) & (df_temp['day_of_month'] ==int(item[2]))
        
    #             df = df_temp[criteria]
    #             # compute process time for all stop points
    #             process_time= 15*len(df) 
                
    #             for index, row in df.iterrows():
    #                 current_row = row
    #                 next_row_index = index + 1
                
    #                 # Check if it's the last row
    #                 if next_row_index < len(df):
    #                     next_row = df.iloc[next_row_index]
    #                 if current_row['sensor']!=next_row['sensor']    : 
    #                     transition_time += dqn.compute_reach_time(current_row['sensor'], next_row['sensor'])
    #             total_time = process_time + transition_time
    #             output[str(item)] = [total_time,transition_time,process_time]
    #             total_process_time+=process_time
    #             total_transition_time+= transition_time
    #             total_total_time+= total_time
        
        
        
        
    #     print('path   :   ' +path )
    #     print('process time: '+ str(total_process_time/60) + '  hour')
    #     print('transition time :' + str(total_transition_time/60))
    #     print('total time :' + str((total_total_time + process_time)/60))
             
        
        
        



