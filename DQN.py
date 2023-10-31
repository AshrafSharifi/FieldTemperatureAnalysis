import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from functions import *


class DQN:
    def __init__(self):
        
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
        self.temperature_weight = 0.7  # Weight for maximizing temperature change
        self.time_weight = 0.3  # Weight for minimizing time
        
        self.memory = deque(maxlen=self.memory_size)
        self.model = self._build_model()
        self.func= functions()
        
        self.reach_time = self.func.time_to_reach_POI()
        data = self.func.get_all_data()
        self.sensorsdata= data
        self.reach_time_minutes = 0
        self.min_time = 15
        self.max_time = 90
        
        fin_name = 'data/temp_info' 
        with open(fin_name, 'rb') as fin:
            temp_info = pickle.load(fin)
            
        fin.close() 
        
        self.min_temp = temp_info['min_temp']
        self.max_temp = temp_info['max_temp']
        


    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         while True:
    #             action = random.randrange(1, self.action_dim)
    #             if action != state[0,0] and action != 0:
    #                 return action
    #     else:
    #         while True:
    #             action = np.argmax(self.model.predict(state)[0])
    #             if action != state[0,0] and action != 0:
    #                 return action
    #             else:
    #                 while True:
    #                     action = random.randrange(1, self.action_dim)
    #                     if action != state[0,0] and action != 0:
    #                         return action
    


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
    
    def step(self,state,action):
        
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
            next_hour, next_min = self.func.add_minutes(base_time, reach_to_next_time)
            new_state = [action,state[0,1],state[0,2],state[0,3],next_hour,next_min]
        return new_state
        
        
    def calculate_reward(self,state, next_state):
        
        
        

        if next_state[0,0]== state[0,0]:
            return -100000000000
        else:    
            
            temp_data =  self.sensorsdata["sensor"+str(state[0,0])]
            temp_data = [value for value in temp_data.values()][0]
            state_data = [value for value in temp_data.values() if value['year'] == state[0,1] and value['month'] == state[0,2] and value['day'] == state[0,3]] 
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
        
        
            temp_data =  self.sensorsdata["sensor"+str(next_state[0,0])]
            temp_data = [value for value in temp_data.values()][0]
            next_state_data = [value for value in temp_data.values() if value['year'] == next_state[0,1] and value['month'] == next_state[0,2] and value['day'] == next_state[0,3]] 
            temp_var = next_state_data[0]['sensorData']
            current_sate = temp_var[temp_var['hour'] == next_state[0,4]]
            next_temperature = 0
            # Check the value of A and select the corresponding row
            if next_state[0,5] == 0:
                next_temperature = current_sate.iloc[0].temp_to_estimate
            elif next_state[0,5] == 15:
                next_temperature = current_sate.iloc[1].temp_to_estimate
            elif next_state[0,5] == 30:
                next_temperature = current_sate.iloc[2].temp_to_estimate
            elif next_state[0,5] == 45:
                next_temperature = current_sate.iloc[3].temp_to_estimate
            else:
                next_temperature = None
                
            # Define a weight for each factor
    
            
            # Calculate the temperature change from current to next state
            temperature_difference = abs(next_temperature - current_temperature)
            
            temperature_change = (temperature_difference - self.min_temp) / (self.max_temp - self.min_temp)
            time_factor = (self.reach_time_minutes - self.min_time) / (self.max_time - self.min_time)
    
            # Combine the two factors with the defined weights
            reward = (self.temperature_weight * temperature_change) - (self.time_weight * time_factor)
            
            
            return reward

        
        
    # def replay(self):
    #     if len(self.memory) < self.batch_size:
    #         return
    #     minibatch = random.sample(self.memory, self.batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         # if not done:
    #         #     target = (reward + gamma * np.amax(self.model.predict(next_state)[0]))
    #         target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    #         self.epsilon = max(self.epsilon, self.epsilon_min)
    
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
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)
            iteration += 1
    
