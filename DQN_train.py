import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from DQN import *
from tensorflow.keras.models import load_model






if __name__ == '__main__':
    # Initialize the DQN agent
    max_steps_per_episode = 50
    episode_count = 20
    train =   False      
    test =     False
    
    initial_state = [1 ,2021, 8, 27, 12, 15]  # Sensor number, year, month, day, hour, minute
    dqn = DQN()
    dqn.get_min_max_temp(initial_state)
    
    if train:
        
       
        # Training loop
        for episode in range(episode_count):  # Adjust the number of episodes as needed
        
        
            state = initial_state  # Sensor number, year, month, day, hour, minute
            state = np.reshape(state, [1, dqn.state_dim])  # Reshape the state for Keras
            
            total_reward = 0
            print("Episode:" + str(episode))
            for step in range(max_steps_per_episode):
                
                
          
                # Choose the sensor to go
                action = dqn.act(state) 
                
                repeat = 0
                while action==state[0,0]:
                    action = dqn.act(state)  
                    repeat += 1
                    if repeat > 5:
                        action = random.randrange(0, dqn.action_dim-1)+1
                    
                # while action == state[0, 0]:
                #     if repeat >= 5:
                #         action = random.randrange(0, dqn.action_dim-1)+1  # Correct the random range
                #     else:
                #         action = dqn.act(state)  # Get a new action from dqn
                #     repeat += 1
                    
                #     if repeat >= 10:
                #         continue
                  
                
                
                # set choosed sensor as next step
                next_state = dqn.step(state,action)  # Update the state based on the robot's movement
                next_state = np.reshape(next_state, [1, dqn.state_dim])
                                  
                    
                reward,temperature_difference,reach_time_minutes = dqn.calculate_reward(state, next_state)  # Calculate the reward
                total_reward += reward
                
                
                # dqn.remember(state, action, reward, next_state, done)
                dqn.remember(state, action, reward, next_state, False)
                state = next_state
                
                # if done:
                #     break
            
                if step >= max_steps_per_episode - 1:
                   break
         
            dqn.replay(20)  # Train the DQN
            
            if episode == 50:
                dqn.model.save("data/dqn_model50.h5")
            elif episode == 70:
                dqn.model.save("data/dqn_model70.h5")  
            elif episode == 80:
                dqn.model.save("data/dqn_model80.h5")                  
    
        # Print the total reward for the episode
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        dqn.model.save("data/dqn_model.h5")
    
    else:
        # Load the DQN model
        loaded_model = load_model("data/dqn_model70 (1).h5")
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
            # Use the loaded model to make a prediction
            # action = np.argmax(temp)+1
            # notdone = True
            # while notdone:
            #     action = np.argmax(temp)+1
            #     if action == state[0,0] or path[str(action)][0] != 'NotVisited':
            #         max_value = max(abs(temp))
            #         temp[action-1] = -max_value
            #         keys = {key for key, value in path.items() if value[0] == 'NotVisited'}
            #         if len(keys) == 0:
            #             finish = True 
            #             continue
            #     else:
            #         notdone = False
            action = int(max_item[0])        
            path[str(action)][0] = 'Visited' 
            # Set the POI which are located befor the current state and not selected by algorithm to passed
            passedkeys = dict()
            passedkeys = {key for key, value in path.items() if value[1] < path[str(action)][1] and value[0] == 'NotVisited'}
            for key in passedkeys:        
                path[key][0] = 'Passed'
            
            next_state = dqn.step(state,action) 
            next_state = np.reshape(next_state, [1, dqn.state_dim])
            reward,temperature_difference,reach_time = dqn.calculate_reward(state, next_state)  # Calculate the reward
            path[str(action)][2] = reward
            path[str(action)][3] = temperature_difference
            path[str(action)][4] = reach_time 
            path[str(action)][5] =  next_state
            state = next_state
            keys = {key for key, value in path.items() if value[0] == 'NotVisited'}
            if len(keys) == 0:
                finish = True 
                
        dqn.print_traj(path)
