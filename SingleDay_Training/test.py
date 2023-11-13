from tensorflow.keras.models import load_model

# Load the DQN model
loaded_model = load_model("dqn_model.h5")
# New state for prediction (make sure it matches the input shape of the model)
new_state = np.array([1, 2021, 8, 26, 13, 30])  # Adjust this state as needed

# Reshape the new state if needed
new_state = np.reshape(new_state, (1, state_dim))

# Use the loaded model to make a prediction
action = np.argmax(loaded_model.predict(new_state)[0])
