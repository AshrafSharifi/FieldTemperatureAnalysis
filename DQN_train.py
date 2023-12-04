import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from DQN import *
from tensorflow.keras.models import load_model







def train(initial_state, path_to_save,state_dict = None):
    dqn = DQN()
    dqn.get_min_max_temp(initial_state)
    # Training loop
    for episode in range(dqn.episode_count):  # Adjust the number of episodes as needed
    
    
        state = initial_state  # Sensor number, year, month, day, hour, minute
        if episode > 1:
            
            state[0] =random.choice(range(1,8))
            keys = list(state_dict.keys())
            random_key = random.choice(keys)
            [y,m] = random_key.split('_')
            state[1]=int(y)
            state[2]=int(m)
            state[3] = random.choice(state_dict[random_key])
            random_hour = random.randint(0, 23)
            random_min = random.choice([0, 15, 30, 45])
            state[4] = random_hour
            state[5] = random_min
        state = np.reshape(state, [1, dqn.state_dim])  # Reshape the state for Keras
        
  
            
        total_reward = 0
        print("Episode:" + str(episode))
        print(state)
        for step in range(dqn.max_steps_per_episode):
            
            
      
            # Choose the sensor to go
            action = dqn.act(state) 
            
            repeat = 0
            while action==state[0,0]:
                action = dqn.act(state)  
                repeat += 1
                if repeat > 5:
                    action = random.randrange(0, dqn.action_dim-1)+1
                

            # set choosed sensor as next step
            next_state,flag = dqn.step(state,action)  # Update the state based on the robot's movement
            next_state = np.reshape(next_state, [1, dqn.state_dim])
                              
                
            reward,temperature_difference,reach_time_minutes = dqn.calculate_reward(state, next_state)  # Calculate the reward
            total_reward += reward
            
            
            # dqn.remember(state, action, reward, next_state, done)
            dqn.remember(state, action, reward, next_state, False)
            state = next_state
            
            # if done:
            #     break
        
            if step >= dqn.max_steps_per_episode - 1:
               break
     
        dqn.replay(dqn.replay_count)  # Train the DQN
        
        # if episode == 50:
        #     dqn.model.save("data/dqn_model50.h5")
        # elif episode == 70:
        #     dqn.model.save("data/dqn_model70.h5")  
        # elif episode == 80:
        #     dqn.model.save("data/dqn_model80.h5")                  

    # Print the total reward for the episode
    print(f"Episode: {episode}, Total Reward: {total_reward}")
    dqn.model.save(path_to_save)
    



def test(initial_state,path_to_save,print_flag=True,dqn = None):
    
    if dqn==None:
        dqn = DQN()
        dqn.get_min_max_temp(initial_state)
    # Load the DQN model
    loaded_model = load_model(path_to_save)
    # New state for prediction (make sure it matches the input shape of the model)
    state = np.array(initial_state)  # Adjust this state as needed
    state = np.reshape(state, [1, dqn.state_dim])  # Reshape the state for Keras
    path = {
        "1": ["NotVisited", 1, 0, 0, 0, np.empty((0, 0)),0],    #POI number: ['status','priority(based on article)'
        "2": ["NotVisited", 2, 0, 0, 0, np.empty((0, 0)),0],    #             'reward','temp difference', 'time to reach', state
        "4": ["NotVisited", 3, 0, 0, 0, np.empty((0, 0)),0],    #             'temperature']
        "6": ["NotVisited", 4, 0, 0, 0, np.empty((0, 0)),0],
        "7": ["NotVisited", 5, 0, 0, 0, np.empty((0, 0)),0],
        "5": ["NotVisited", 6, 0, 0, 0, np.empty((0, 0)),0],
        "3": ["NotVisited", 7, 0, 0, 0, np.empty((0, 0)),0]
           }
    path[str(state[0,0])][0] = 'Visited' 
    
    path[str(state[0,0])][5] =  state
    initial_day = int(initial_state[3])
    finish = False
    while finish == False:
        # Reshape the new state if needed
        
        temp = loaded_model.predict(state)[0]
        temp = np.reshape(temp, [1, dqn.action_dim])[0]
        
        path_nodes = dict()
        POI = 1
        for item in temp:
            path_nodes[str(POI)] = [path[str(POI)][0],item]
            POI += 1
            
            
        max_value = float('-inf')
        max_item = None
        
        for key, value in path_nodes.items():
            if key != state[0,0] and value[0] == 'NotVisited' and value[1] > max_value:
                max_value = value[1]
                max_item = (key, value)

        action = int(max_item[0])        
        path[str(action)][0] = 'Visited' 
        # Set the POI which are located befor the current state and not selected by algorithm to passed
        passedkeys = list()
        
        
        
        current_key = path[str(state[0][0])][1]+1
        end_key = path[str(action)][1]
        while current_key != end_key:
            
            if(int(current_key)==8):
                current_key=1
            p_k = [key for key, value in path.items() if value[1] == int(current_key) and value[0] == 'NotVisited']
            if len(p_k)!=0:
                passedkeys.append(p_k)
                path[p_k[0]][0] = 'Passed'
                next_key = path[p_k[0]][1]+1
                current_key = int(next_key)
            else:
                break
            
      
      
        # passedkeys = {key for key, value in path.items() if value[1] < path[str(action)][1] and value[0] == 'NotVisited'}
        # for key in passedkeys:        
        #     path[key][0] = 'Passed'
       
        # if state[0][0]<action:
        #    passedkeys.add({key for key, value in path.items() if value[1] > path[str(state[0][0])][1] and value[0] == 'NotVisited'})
        
        next_state,Flag = dqn.step(state,action) 
        next_state = np.reshape(next_state, [1, dqn.state_dim])
        reward,temperature_difference,reach_time = dqn.calculate_reward(state, next_state)  # Calculate the reward
        path[str(action)][2] = reward
        path[str(action)][3] = temperature_difference
        path[str(action)][4] = reach_time 
        path[str(action)][5] =  next_state
        state = next_state
        keys = {key for key, value in path.items() if value[0] == 'NotVisited'}
        if int(state[0][4]) == 22:
            A=1
        # if len(keys) == 0 or int(state[0][4]) == 0:
        if len(keys) == 0 or Flag:
            finish = True
            if Flag:
                state[0][3]=state[0][3]+1
                path[str(action)][5] =  state
        
            
             
    
    
    table = dqn.print_traj(path,print_flag)
    
    
    
    return (table[-1][6])[0],path
        
        
    
    

    




if __name__ == '__main__':
    # Initialize the DQN agent

    train_phase =   False      
    initial_state =  [1,2021, 9, 28, 9, 00] 
    # initial_state =  [1 ,2021, 10, 5, 21, 00] 

    # Sensor number, year, month, day, hour, minute
    
    path_to_save = "data/current_work/DQN_models/With_process_time/dqn_model80.h5"
    # path_to_save = "data/models/dqn_model160_old.h5"
   
   
    if train_phase: 
        train(initial_state,path_to_save)
    else:
        test(initial_state,path_to_save)
       
