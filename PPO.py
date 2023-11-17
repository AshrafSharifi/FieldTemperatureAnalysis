import numpy as np
import tensorflow as tf
import gym





if __name__ == "__main__":
    
    # Define the environment and hyperparameters
    env = gym.make('CartPole-v1')  # Replace with your desired environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    lr_actor = 0.001
    lr_critic = 0.001
    gamma = 0.99
    clip_epsilon = 0.2
    epochs = 10
    timesteps_per_batch = 2048
    # Define the actor and critic networks
    actor = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_dim, activation='softmax')
    ])
    
    critic = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(state_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    actor_optimizer = tf.optimizers.Adam(learning_rate=lr_actor)
    critic_optimizer = tf.optimizers.Adam(learning_rate=lr_critic)
    
    # PPO training loop
    for epoch in range(epochs):
        observations, actions, advantages, returns = [], [], [], []
        observation = env.reset()
        observation = observation[0]
        done = False
        t = 0
    
        while True:
                
            inputval = np.reshape(observation, [1,state_dim])
            action_prob = (actor.predict(inputval)[0])
         
            # action = np.random.choice(action_dim, p=action_prob)
            action = np.argmax(action_prob)

            ReturnVal = env.step(action)
            next_observation = ReturnVal[0] 
            reward = ReturnVal[1]
            done = ReturnVal[2]
            observations.append(observation)
            actions.append(action)
            reward = np.clip(reward, -1.0, 1.0)  # Clip rewards
            advantages.append(reward)
            returns.append(reward + (1 - done) * gamma * critic.predict(np.reshape(next_observation, [1, state_dim]))[0])
    
            if t == timesteps_per_batch:
                break
    
            observation = next_observation
            t += 1
    
        # Update actor and critic networks
        observations = np.array(observations)
        actions = np.array(actions)
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        
        for _ in range(5):  # Number of PPO epochs
            with tf.GradientTape() as tape:
                value_prediction = critic(observations)
                value_loss = tf.reduce_mean(tf.square(returns - value_prediction))
            critic_gradients = tape.gradient(value_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))
            
            
        for _ in range(5):  # Number of PPO epochs
            with tf.GradientTape() as tape:
                action_prob = actor(observations)
                action_masks = tf.one_hot(actions, action_dim)
                selected_action_prob = tf.reduce_sum(action_prob * action_masks, axis=1)
                old_action_prob = tf.stop_gradient(selected_action_prob)
                
                ratio = (np.array(action_prob))[:,np.array(actions)] / (old_action_prob + 1e-10)
                surrogate_obj1 = ratio * advantages
                surrogate_obj2 = tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                surrogate_loss = -tf.reduce_mean(tf.minimum(surrogate_obj1, surrogate_obj2))
            actor_gradients = tape.gradient(surrogate_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
    

