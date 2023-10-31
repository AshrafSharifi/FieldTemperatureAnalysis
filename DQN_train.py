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
    episode_count = 100
    train = True      
    test =    False
    dqn = DQN()
    
    if train:
        
       
        # Training loop
        for episode in range(episode_count):  # Adjust the number of episodes as needed
        
        
            state = [1 ,2021, 8, 27, 12, 15]  # Sensor number, year, month, day, hour, minute
            state = np.reshape(state, [1, dqn.state_dim])  # Reshape the state for Keras
            total_reward = 0
            print("Episode:" + str(episode))
            for step in range(max_steps_per_episode):
                
                
          
                # Choose the sensor to go
                action = dqn.act(state) 
               

                
                # set choosed sensor as next step
                next_state = dqn.step(state,action)  # Update the state based on the robot's movement
                next_state = np.reshape(next_state, [1, dqn.state_dim])
    
                reward = dqn.calculate_reward(state, next_state)  # Calculate the reward
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
                dqn.model.save("dqn_model50.h5")
            elif episode == 70:
                dqn.model.save("dqn_model70.h5")     
    
        # Print the total reward for the episode
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        dqn.model.save("dqn_model.h5")
    
    else:
        # Load the DQN model
        loaded_model = load_model("data/dqn_model50.h5")
        # New state for prediction (make sure it matches the input shape of the model)
        state = np.array([1 ,2021, 8, 27, 12, 15])  # Adjust this state as needed

        # Reshape the new state if needed
        state = np.reshape(state, [1, dqn.state_dim])  # Reshape the state for Keras
        temp = loaded_model.predict(state)[0]
        # Use the loaded model to make a prediction
        action = np.argmax(temp)+1
        
        next_state = dqn.step(state,action) 
        next_state = np.reshape(next_state, [1, dqn.state_dim])
        reward = dqn.calculate_reward(state, next_state)  # Calculate the reward

