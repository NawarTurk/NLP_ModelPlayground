from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





# Generate random synthetic data for training the model
num_samples = 1000  # Number of data points (rows)
num_features = 8    # Number of features (columns, matches input_shape)
num_sample_valid = 200

# Generate features
X_train = np.random.rand(num_samples, num_features)  # Each feature value is randomly generated between 0 and 1
X_valid = np.random.rand(num_sample_valid, num_features)
# Randomly generate binary labels (0 or 1)
y_train = np.random.randint(0, 2, size=num_samples)  # Randomly generate 0 or 1 for each sample
y_valid = np.random.randint(0, 2, size=num_sample_valid)  # Randomly generate 0 or 1 for each validation sample


model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[num_features]),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()


print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))