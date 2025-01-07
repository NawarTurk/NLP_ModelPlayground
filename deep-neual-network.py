from tensorflow import keras 
from tensorflow.keras import layers 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Title: Creating a Neural Network with Three Hidden Layers in TensorFlow

# Define the model using the Sequential API
# This approach simplifies model creation by stacking layers sequentially.
model = keras.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=[8]),  # Input layer with 8 features
    layers.Dense(units=512, activation='relu'),  # First hidden layer with ReLU activation
    layers.Dense(units=512, activation='relu'),  # Second hidden layer with ReLU activation
    layers.Dense(units=1)  # Output layer with 1 unit (no activation for regression tasks)
])

'''
# Alternatively, explicitly separating dense layers and activation functions can provide flexibility:
# This is useful when needing additional control or introducing custom layers or functionality.
model = keras.Sequential([
    layers.Dense(units=512, input_shape=[8]),  # Input layer without activation
    layers.Activation('relu'),  # ReLU activation function applied separately
    layers.Dense(units=512),  # First hidden layer without activation
    layers.Activation('relu'),
    layers.Dense(units=512),  # Second hidden layer without activation
    layers.Activation('relu'),
    layers.Dense(units=1)  # Output layer (linear activation for regression tasks)
])
'''

# Compile the model
# Optimizer: Adam for adaptive learning rates
# Loss: Mean Absolute Error (MAE) for regression tasks
model.compile(
    optimizer='adam',
    loss='mae'
)

# Generate random synthetic data for training the model
num_samples = 1000  # Number of data points (rows)
num_features = 8    # Number of features (columns, matches input_shape)
X = np.random.rand(num_samples, num_features)  # Each feature value is randomly generated between 0 and 1
y = np.sum(X, axis=1) + np.random.normal(scale=0.1, size=num_samples) # y is a linear combination of features with some added Gaussian noise for realism

# Train the model
history = model.fit(
    X, y,
    batch_size=128,  # Process 128 samples at a time  # Batch size: Number of samples processed before updating the model
    epochs=50  # Train the model for 200 epochs # Epochs: Number of complete passes through the training dataset
)

# Convert the training history to a Pandas DataFrame for visualization
history_df = pd.DataFrame(history.history) # The history object contains the loss values for each epoch

# Plot the training loss over epochs to observe learning progression
history_df.plot(title="Training Loss Over Epochs")
plt.xlabel("Epochs")  # Optional: Label for x-axis
plt.ylabel("Loss")    # Optional: Label for y-axis
plt.show()
