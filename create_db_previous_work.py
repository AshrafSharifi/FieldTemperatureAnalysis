from functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import csv
import pickle
import DQN_train as dqn_train
import copy
from DQN import *
        
    

def create_states_dict():
    

    pd.options.mode.chained_assignment = None 
    
    sensorList = list(range(1,9))
    data_size = 0
    states_dict = dict()
    for sensor in sensorList:
        csv_path = "data/sensor_t" + str(sensor) + ".csv"
        sensor_data = pd.read_csv(csv_path)
        
        # -------remove data of days which has small size
        grouped = sensor_data.groupby(['year','month','day_of_month'])
        group_sizes = grouped.size().reset_index(name='GroupSize')
        valid_groups = grouped.filter(lambda x: len(x) ==96)
        indices_to_keep = valid_groups.index
        sensor_data = sensor_data.loc[indices_to_keep]
        sensor_data['timestamp'] = pd.to_datetime(sensor_data['complete_timestamp(YYYY_M_DD_HH_M)'], format='%Y_%m_%d_%H_%M')
        sensor_data = sensor_data.sort_values(by='timestamp')
        sensor_data.to_csv(csv_path, index=False, encoding='utf-8')
        # ----------------------------------------------

        
        sensor_dict = dict()
        output = dict()
        sensor_dict["years"] = (sensor_data['year'].unique())
        num_year = sensor_dict["years"].shape[0]
        
        for year in sensor_dict["years"]:
            # Filter data based on the specified date
            
            idx_y = (sensor_data['year'] == year)
            filtered_data_step1 = sensor_data.loc[idx_y,:]

            sensor_dict["months"] = (filtered_data_step1['month'].unique())
            num_month = sensor_dict["months"].shape[0]
            
            for month in sensor_dict["months"]:
                
                idx_m = (filtered_data_step1['month'] == month) 
                filtered_data_step2 = filtered_data_step1.loc[idx_m,:]
                
        


                sensor_dict["days"] = (filtered_data_step2['day_of_month'].unique())
                
                num_day = (filtered_data_step2['day_of_month'].unique()).shape[0]
                
                output[str(year)+'_'+str(month)]=sensor_dict["days"]
                data_size += num_day
                for day in sensor_dict["days"]:
                    
                    idx_d = (filtered_data_step2['day_of_month'] == day)
                    filtered_data = filtered_data_step2.loc[idx_d,:]
                    

                     

        # print((num_year * num_month * num_day))
        states_dict[sensor] = sensor_dict 
        with open('data/states_dict', 'wb') as fout:
            pickle.dump(output, fout)
        fout.close()
def add_status_col_to_csv():
    
    pd.options.mode.chained_assignment = None 
    
    # Define a threshold to detect sharp changes
    # threshold = 2.69

    sensorList = list(range(1,9))
    for sensor in sensorList:
        
        # Specify the CSV file path
        csv_file_path = pd.read_csv("data/sensor_t" + str(sensor) + ".csv")
        
        # Specify the input CSV file and output CSV file
        input_csv_file =  "data/sensor_t" + str(sensor) + ".csv"
        output_csv_file = "data/DB/DB_with_status/sensor_t" + str(sensor) + ".csv"
        del_csv_file = "data/DB/DB_with_deleted_rows/sensor_t" + str(sensor) + ".csv"
        
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_file)
        
        # Define the default value for the new column
        default_value = 'Not_Visited'
        
        # Add a new column with the default value to the DataFrame
        df['status'] = default_value
        df.to_csv(output_csv_file, index=False, encoding='utf-8')
        df.to_csv(del_csv_file, index=False, encoding='utf-8')
    
def localize_row(df,state):
    
    all_conditions = (df['hour'] == state[4]) & (df['month'] == state[2]) & (df['day_of_month'] == state[3])
    # Check the value of A and select the corresponding row
    if state[5] == 0:
        condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '0')
    elif state[5] == 15:
        condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '1')
    elif state[5] == 30:
        condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '2')
    elif state[5] == 45:
        condition = (df['complete_timestamp(YYYY_M_DD_HH_M)'].str[-1] == '3')
    
    indx = (df[all_conditions & condition].index)
    if indx.size == 1:
        return indx[0]
    else:
        return -1


def del_rows(del_sensors,state):
    for us in del_sensors:
        delete_csv_file = "data/DB/DB_with_deleted_rows/sensor_t" + str(us) + ".csv"
        us_df = pd.read_csv(delete_csv_file)
        us_state = state.copy()
        us_state[0] = us
        us_indx = localize_row(us_df, us_state)
        if us_indx !=-1:
            us_df = us_df.drop(us_indx)
            us_df.to_csv(delete_csv_file, index=False, encoding='utf-8')
                
                
def add_row(state,df_train,df_test):

    sensor = state[0]
    csv_file = "data/sensor_t" + str(sensor) + ".csv"
    
    df = pd.read_csv(csv_file)
    
    indx = localize_row(df,state)
  
    df_train.loc[len(df_train)] = list(df.iloc[indx]) + [str(sensor)]
    del_sensors = list(range(1,9))
    del_sensors.remove(sensor)
    
    for item in del_sensors:
        csv_file = "data/sensor_t" + str(item) + ".csv"
        df = pd.read_csv(csv_file)
        state[0] = item
        indx = localize_row(df,state)
        df_test.loc[len(df_test)] = list(df.iloc[indx]) + [str(item)]
        

    return df_train,df_test
            
        
def find_next_item(arr, num):
    # Find the index of the given number in the array
    try:
        index = arr.index(num)
    except ValueError:
        print(f"The number {num} is not in the array.")
        return None

    # Calculate the index of the next number in the circular array
    next_index = (index + 1) % len(arr)

    # Return the next number
    return arr[next_index]    
       
        
if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None 

    fin_name = 'data/states_dict'
    with open(fin_name, 'rb') as fin:
        states_dict = pickle.load(fin)
    fin.close() 
    
    change_month = False
    state_dim = 6
    csv_file = "data/sensor_t1.csv"   
    df = pd.read_csv(csv_file)
    df_train =  pd.DataFrame(columns=list(df.columns) + ['sensor'])
    df_test =  pd.DataFrame(columns=list(df.columns) + ['sensor'])
    func= functions()
    path = [1,2,4,6,7,5,3]

    prev_time_data=[]
    process_counter=0
    process_in_day=[]
    for key,value in states_dict.items():

        [y,m] = key.split('_')
        d = value[0]
        h = 0
        Min = 0
        change_month=False
        while change_month==False:

            current_state = [1 ,int(y), int(m), d, h, Min]  # Sensor number, year, month, day, hour, minute   
            while True:
                print(current_state)
                prev_time_data.append(current_state)
                df_train,df_test = add_row(current_state.copy(),df_train,df_test)
                
                base_time = str(current_state[4]) + ':' + str(current_state[5]) + ':00'
                next_hour, next_min, Flag = func.add_minutes(base_time, 15)
                
                next_sensor = find_next_item(path,current_state[0])
                current_state = np.array([int(next_sensor), int(current_state[1]), int(current_state[2]), int(current_state[3]), next_hour, next_min])
                # current_state = np.reshape(state, [1, state_dim])
                prev_time_data.append(current_state)
                if Flag:
                    current_state[3]=current_state[3]+1
                process_counter +=1    
                if int(current_state[3]) == d+1:
                    process_in_day.append(process_counter)
                    process_counter=0
                    d = current_state[3]
                    m = current_state[2]
                    y = current_state[1]
                    temp_val = states_dict[str(y)+'_'+str(m)]
                    if d not in temp_val:
                        while d not in temp_val:
                            d += 1
                            if(d>max(temp_val)):
                                change_month = True
                                break
                            
                    h = current_state[4]
                    Min = current_state[5]
                    break
    with open('data/prev_all_traj', 'wb') as fout:
        pickle.dump(prev_time_data, fout)
    fout.close()   
    with open('data/prev_process_in_day', 'wb') as fout:
        pickle.dump(process_in_day, fout)
    fout.close()            
    # df_train.to_csv("data/DB/DB_previous_work/train_DB.csv", index=False, encoding='utf-8')
    # df_test.to_csv("data/DB/DB_previous_work/test_DB.csv", index=False, encoding='utf-8')
                    
                
 
    


                    
                
    
    
    
    
    
    
    
    