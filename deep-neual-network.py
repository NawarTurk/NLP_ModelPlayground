from tensorflow import keras 
from tensorflow.keras import layers, callbacks
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
model.compile(
    optimizer='adam',  # Optimizer: Adam for adaptive learning rates
    loss='mae'  # Loss: Mean Absolute Error (MAE) for regression tasks
)

early_stopping = callbacks.EarlyStopping(
    min_delta= 0.001,
    patience= 20, # If there hasn't been at least an improvement of 0.001 in the validation loss over the previous 20 epochs, 
                  # then stop the training and keep the best model you found
    restore_best_weights=True  # estore_best_weights=True, the model will revert to the weights from the epoch 
                               # with the lowest validation loss observed  during the entire training process up to that point, not just the last 20 epochs.
)

# Generate random synthetic data for training the model
num_samples = 1000  # Number of data points (rows)
num_features = 8    # Number of features (columns, matches input_shape)
num_sample_valid = 200

X_train = np.random.rand(num_samples, num_features)  # Each feature value is randomly generated between 0 and 1
y_train = np.sum(X_train, axis=1) + np.random.normal(scale=0.1, size=num_samples) # y is a linear combination of features with some added Gaussian noise for realism
X_valid = np.random.rand(num_sample_valid, num_features)
y_valid = np.sum(X_valid, axis=1) + np.random.normal(scale=0.2, size=num_sample_valid)


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,  # Process 128 samples at a time  # Batch size: Number of samples processed before updating the model
    epochs=500,  # Train the model for 200 epochs # Epochs: Number of complete passes through the training dataset
    callbacks=[early_stopping],
    verbose=0  # turn off training log
)

# Convert the training history to a Pandas DataFrame for visualization
history_df = pd.DataFrame(history.history) # The history object contains the loss values for each epoch

# Plot the training loss over epochs to observe learning progression
history_df.loc[:, ['loss', 'val_loss']].plot(title="Training Loss Over Epochs")
print(f'Minimum validation loss: {history_df['val_loss'].min()}')
plt.xlabel("Epochs")  # Optional: Label for x-axis
plt.ylabel("Loss")    # Optional: Label for y-axis
plt.show()

