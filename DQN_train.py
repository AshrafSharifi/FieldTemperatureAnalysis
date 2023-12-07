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
    his_trajectory,initial_state = extract_history_traj(dqn,np.reshape(initial_state, [1, dqn.state_dim]) )
    initial_his_trajectory = his_trajectory.copy()
    
    # Training loop
    for episode in range(dqn.episode_count):  # Adjust the number of episodes as needed
    
    
        state = initial_state.copy()  # Sensor number, year, month, day, hour, minute
        his_trajectory = initial_his_trajectory.copy()
        if episode > 1 and state_dict!=None:
            
            state[0,0] =random.choice(range(1,8))
            keys = list(state_dict.keys())
            random_key = random.choice(keys)
            [y,m] = random_key.split('_')
            state[0,1]=int(y)
            state[0,2]=int(m)
            state[0,3] = random.choice(state_dict[random_key])
            random_hour = random.randint(0, 23)
            random_min = random.choice([0, 15, 30, 45])
            state[0,4] = random_hour
            state[0,5] = random_min
            his_trajectory,state = extract_history_traj(dqn,state)
            
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
                              
                
            reward,temperature_difference,reach_time_minutes,his_trajectory = dqn.calculate_reward(state, next_state,his_trajectory)  # Calculate the reward
            
            total_reward += reward
            
            
            # dqn.remember(state, action, reward, next_state, done)
            dqn.remember(state, action, reward, next_state, False)
            state = next_state
            
            # if done:
            #     break
        
            if step >= dqn.max_steps_per_episode - 1:
               break
     
        dqn.replay(dqn.replay_count)  # Train the DQN
        
        if episode %10 == 0:
           dqn.model.save("data/current_work_with_process_time/DQN_models/With_process_time/dqn_model"+str(episode)+".h5")
        # elif episode == 70:
        #     dqn.model.save("data/dqn_model70.h5")  
        # elif episode == 80:
        #     dqn.model.save("data/dqn_model80.h5")                  

    # Print the total reward for the episode
    print(f"Episode: {episode}, Total Reward: {total_reward}")
    dqn.model.save(path_to_save)
    



def test(initial_state,path_to_save,print_flag=True,dqn = None,his_traj=None):
    
    if dqn==None:
        dqn = DQN()
        dqn.get_min_max_temp(initial_state)
    his_trajectory=his_traj  
    if his_trajectory==None:
        his_trajectory,initial_state = extract_history_traj(dqn,np.reshape(initial_state, [1, dqn.state_dim]))
   
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
    path[str(state[0,0])][5] = state
    his_trajectory[str(state[0,0])][0]=state
    temperature_difference = abs(his_trajectory[str(state[0,0])][1] - dqn.get_current_temp(state))
    path[str(state[0,0])][3] =  temperature_difference
    initial_day = int(initial_state[0,3])
    finish = False
    num_of_visited_POIs = 0
    while finish == False:
        # Reshape the new state if neededhisory_trajectory=his_trajectory
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
        p_key = dqn.get_previous_key(path, str(action))
        is_visited=1
        next_state,Flag = dqn.step(state,action,1) 
        
        next_state = np.reshape(next_state, [1, dqn.state_dim])
        reward,temperature_difference,reach_time,his_trajectory = dqn.calculate_reward(state, next_state,his_trajectory)  # Calculate the reward
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
        
            
             
         
    
    table = dqn.print_traj(path,print_flag,his_trajectory)
    
    
    last_state = (table[-1][6])
    # if path[last_state[0,0]]=='Visited':
    current_sensor = last_state[0,0]
    current_hour = last_state[0,4]
    current_minute = last_state[0,5]
    base_time = str(current_hour)+':'+str(current_minute)+':00'
    next_hour, next_min, Flag = dqn.func.add_minutes(base_time, dqn.process_time+15 )
    
    next_state = dqn.find_next_item(last_state[0,0])
    if Flag:
        state[0,3]=state[0,3]+1
        
    new_state = [next_state,state[0,1],state[0,2],state[0,3],next_hour,next_min]
        
    
    return np.reshape(new_state, [1, dqn.state_dim]),path,his_trajectory,table[-1][6]
        
        
    
    

    
def extract_history_traj(dqn,current_state):
    current_sensor = current_state[0,0]
    next_sensor=0
    path = [1,2,4,6,7,5,3]
    history_path = dict()
    initial_sensor=current_state[0,0]

    while initial_sensor != next_sensor:
        current_state = np.reshape(current_state, [1, dqn.state_dim])
        current_sensor = current_state[0,0]
        temperature = dqn.get_current_temp(current_state)
        history_path[str(current_sensor)]=[current_state, temperature]
        base_time = str(current_state[0,4]) + ':' + str(current_state[0,5]) + ':00'
        next_hour, next_min, Flag = dqn.func.add_minutes(base_time, 15)
        next_sensor = find_next_item(path,current_state[0,0])
        current_state = np.array([int(next_sensor), int(current_state[0,1]), int(current_state[0,2]), int(current_state[0,3]), next_hour, next_min])
    
    # Custom sorting key based on day, hour, and minutes
    def sorting_key(item):
        return tuple(item[0][0, 2:5])
    
    # Sort the dictionary based on the custom key
    his_trajectory = dict(sorted(history_path.items(), key=lambda x: sorting_key(x[1])))
    last_item = list(his_trajectory.values())[-1]
    current_state = last_item[0]
    base_time = str(current_state[0,4]) + ':' + str(current_state[0,5]) + ':00'
    next_hour, next_min, Flag = dqn.func.add_minutes(base_time, 15)
    next_sensor = find_next_item(path,current_state[0,0])
    current_state = np.array([int(next_sensor), int(current_state[0,1]), int(current_state[0,2]), int(current_state[0,3]), next_hour, next_min])

    return history_path,np.reshape(current_state, [1, dqn.state_dim])


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
    # Initialize the DQN agent
    
    train_phase =   False      
    initial_state =  [1,2021, 9, 29, 11, 0] 
    # initial_state =  [1 ,2021, 10, 5, 21, 00] 

    # Sensor number, year, month, day, hour, minute
    
    path_to_save = "data/current_work_with_process_time/DQN_models/With_process_time/With_normalize/dqn_model160.h5"
    # path_to_save = "data/models/dqn_model160_old.h5"
   
   
    if train_phase: 
        
        train(initial_state,path_to_save)
    else:
        test(initial_state,path_to_save)
       
