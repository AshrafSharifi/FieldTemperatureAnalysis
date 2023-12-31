import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from functions import *
from tabulate import tabulate
from keras.layers import BatchNormalization
class DQN:
    def __init__(self):
        
        self.max_steps_per_episode = 20
        self.episode_count = 200
        self.replay_count = 10
        # Define your state and action spaces, and other relevant parameters
        self.state_dim = 6  # Dimension of the state
        self.action_dim = 7  # Number of possible actions
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.batch_size = 8  # Batch size for training
        self.memory_size = 20000  # Size of the experience replay buffer
        self.reach_time = dict()
        self.temperature_weight = 3  # Weight for maximizing temperature change
        self.time_weight = 1  # Weight for minimizing time
        
        self.memory = deque(maxlen=self.memory_size)
        self.model = self._build_model()
        self.func= functions()
        
        self.reach_time = self.func.time_to_reach_POI()
        data = self.func.get_all_data()
        self.sensorsdata= data
        self.reach_time_minutes = 0
        
        self.min_time = 15
        self.max_time = 90
        
        self.min_temp = 0
        self.max_temp = 0
        self.process_time = 15
        # self.process_penalty = 0.5*(self.process_time)
        self.process_penalty = 0.5*(self.process_time / 60)
        # self.state=None
        # self.get_min_max_temp(self.state)

        
        # fin_name = 'data/temp_info' 
        # with open(fin_name, 'rb') as fin:
        #     temp_info = pickle.load(fin)
            
        # fin.close() 
        
        # self.min_temp = temp_info['min_temp']
        # self.max_temp = temp_info['max_temp']
        
        
 
    def get_min_max_temp(self,state=None):
        # temp_data =  self.sensorsdata["sensor"+str(state[0])]
        # temp_data = [value for value in temp_data.values()][0]
        # state_data = [value for value in temp_data.values() if value['year'] == state[1] and value['month'] == state[2] and value['day'] == state[3]] 
        # temp_var = state_data[0]['sensorData']
        # self.min_temp = min(temp_var["temp_to_estimate"])
        # self.max_temp = max(temp_var["temp_to_estimate"])
        
        fin_name = 'data/temp_info' 
        with open(fin_name, 'rb') as fin:
            temp_info = pickle.load(fin)
        fin.close() 
        
        self.min_temp = temp_info['min_temp']
        self.max_temp = temp_info['max_temp']
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(24, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def find_next_item(self,num):
        arr = [1,2,4,6,7,5,3]
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


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(0, self.action_dim-1)
            return action+1
        else:
            # while True:
            temp = self.model.predict(state)[0]
            action = np.argmax(temp)
            return action+1
                # if action != state[0,0] and action != 0:
                #     return action
                # else:
                #     while True:
                #         action = random.randrange(1, self.action_dim)
                #         return action             
    
    def compute_reach_time(self,current_sensor, next_sensor):
        reach_time_data = self.reach_time["sensor"+str(current_sensor)]
        reach_time = reach_time_data[str(current_sensor)+'_'+str(next_sensor)]
        return reach_time
        
    
    
    def step(self,state,action,is_visited=0):
        
        current_sensor = state[0,0]
        current_hour = state[0,4]
        current_minute = state[0,5]
        reach_time = self.reach_time["sensor"+str(state[0,0])]
        if current_sensor == action:
           new_state = state
        else:
            reach_to_next_time = reach_time[str(current_sensor)+'_'+str(action)]
            self.reach_time_minutes = reach_to_next_time
            base_time = str(current_hour)+':'+str(current_minute)+':00'
            next_hour, next_min, Flag = self.func.add_minutes(base_time, reach_to_next_time+(is_visited*self.process_time) )
            new_state = [action,state[0,1],state[0,2],state[0,3],next_hour,next_min]
        return new_state, Flag
        
    def get_current_temp(self,state):
            temp_data =  self.sensorsdata["sensor"+str(state[0,0])]
            temp_data = [value for value in temp_data.values()][0]
            state_data = [value for value in temp_data.values() if value['year'] == state[0,1] and value['month'] == state[0,2] and value['day'] == state[0,3]] 
            if len(state_data)==0:
                return None
            temp_var = state_data[0]['sensorData']
            current_sate = temp_var[temp_var['hour'] == state[0,4]]
            
            current_temperature = 0
            # Check the value of A and select the corresponding row
            if state[0,5] == 0:
                current_temperature = current_sate.iloc[0].temp_to_estimate
            elif state[0,5] == 15:
                current_temperature = current_sate.iloc[1].temp_to_estimate
            elif state[0,5] == 30:
                current_temperature = current_sate.iloc[2].temp_to_estimate
            elif state[0,5] == 45:
                current_temperature = current_sate.iloc[3].temp_to_estimate
            else:
                current_temperature = None
                
            return current_temperature
    
    def calculate_reward_for_first_state(self,state, temperature_change):
        
       
       
            temperature_change =abs((temperature_change - self.min_temp) / (self.max_temp - self.min_temp))
            # Combine the two factors with the defined weights
            reward = (self.temperature_weight *abs( temperature_change)) - self.process_penalty
            
            
            return reward
    
    def calculate_reward(self,state, next_state,his_trajectory):
        
       
        if next_state[0,0]== state[0,0]:
            return -1000,0,90
        else:   
            
            
                
                    
            desired_item = None

            for key, value in his_trajectory.items():
                if value[0][0, 0] == next_state[0,0]:
                    desired_item = value
                    break
    
           
            next_temperature = self.get_current_temp(next_state)
            his_trajectory[key] = [next_state,next_temperature]

            # Calculate the temperature change from current to next state
            temperature_change1 = next_temperature - desired_item[1]
            
            time_factor = self.reach_time_minutes
            temperature_change =abs((temperature_change1 - self.min_temp) / (self.max_temp - self.min_temp))
            time_factor = abs((self.reach_time_minutes - self.min_time) / (self.max_time - self.min_time))
            
            # Combine the two factors with the defined weights
            reward = (self.temperature_weight *abs( temperature_change)) - (self.time_weight * time_factor) - self.process_penalty
            
            
            return reward,temperature_change1,self.reach_time_minutes,his_trajectory

        

    
    def replay(self, max_iterations):
        
        iteration = 0  # Initialize the iteration counter
        while iteration < max_iterations:
            if len(self.memory) < self.batch_size:
                return
            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                # if action == state[0,0] or state[0,0]== next_state[0,0]:
                #     continue
                target = reward
                temp = self.model.predict(next_state)[0];
                target = (reward + self.gamma * np.amax(temp))
                target_f = self.model.predict(state)
                target_f[0][action-1] = target
                self.model.fit(state, target_f, epochs=10, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            iteration += 1
            
            
    def get_previous_key(self,circular_dict, current_key):
        keys = list(circular_dict.keys())
        index = keys.index(current_key)
        previous_index = (index - 1) % len(keys)
        previous_key = keys[previous_index]
        return previous_key
    
    def has_shape_zero(self,value):
        return value[5].shape == (0, 0)

    def refine_time(self,table):
        
        visited_rows = [0]

        for row in table:
            visited_rows.append(1 if row[1] == 'Visited' else 0)

        cumulative_sum = np.cumsum(visited_rows)  
        
        count = 0
        for row in table:
            if row[1] == 'Passed':
                state = row[6]
                base_time = str(state[0][4])+':'+str(state[0][5])+':00'
                extra_time = self.process_time 
    
                state[0][4], state[0][5], Flag = self.func.add_minutes(base_time, extra_time)
                row[6]=state
                
            count += 1

        initial_day = table[0][6][0][3]   
        for row in table:
            state = row[6]
            if state[0][4] == 0 and state[0][3]==initial_day:  # Check if hour is 0 and minutes is 0
               state[0][3] += 1  # Increment the day by 1
        return table
    
    def add_prev_temp(self,table,his_trajectory):
        

        for row in table:
            if row[1] == 'Passed' and row[-1]!=None:
                temperature_difference = abs(his_trajectory[row[0]][1] - row[-1])
                row[4] = temperature_difference
                
            
        return table
    def print_traj(self,path,print_flag=True,his_trajectory=None):
        
        passedkeys = {key:value[1] for key, value in path.items() if value[0] == 'Passed'}
        list_of_tuples = [(key, value) for key, value in passedkeys.items()]

        # Sort the list of tuples based on the values
        sorted_list = sorted(list_of_tuples, key=lambda x: x[1])
        
        while any(self.has_shape_zero(value) for value in path.values() if value[0]!= 'NotVisited'):
            for (key, val) in sorted_list:
                while path[key][5].shape == (0, 0):
                    previouskey_list = self.get_previous_key(path, key)
                    
                    if not previouskey_list:
                        break  # Break the loop if there is no valid previous key
            
                    previouskey = previouskey_list[0]
            
                    if path[previouskey][5].shape != (0, 0):
                        pre_state = np.reshape(path[previouskey][5], [1, self.state_dim]) 
                        base_time = str(pre_state[0, 4]) + ':' + str(pre_state[0, 5]) + ':00'
                        next_hour, next_min, Flag = self.func.add_minutes(base_time, 15)
                        state = np.array([int(key), int(pre_state[0, 1]), int(pre_state[0, 2]), int(pre_state[0, 3]), next_hour, next_min])
                        path[key][5] = np.reshape(state, [1, self.state_dim])
            
                    key = previouskey

        
        headers = ["POI_number", "Status", "Priority", "Reward", "Temp_difference", "Reach_time", "State","temperature"]
        table = []
        for key, values in path.items():
            if values[0] != 'NotVisited':
                table.append([key, *values])
                
        table = sorted(table, key=lambda x: (x[6][0][3],x[6][0][4], x[6][0][5]))
        table = self.refine_time(table)
        
        table = sorted(table, key=lambda x: (x[6][0][3],x[6][0][4], x[6][0][5]))
        for entry in table:
            if entry[1] != 'NotVisited':
                path_key = entry[1]  # Assuming entry[0] is the key in the path dictionary
                entry[7] = self.get_current_temp(entry[6])
        
        table =self.add_prev_temp(table,his_trajectory)
        if print_flag:
            print(tabulate(table, headers, tablefmt="pretty"))
        return table
        