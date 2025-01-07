import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# List of activation functions to compare
activations = ['relu', 'elu', 'selu', 'swish']

x = tf.linspace(-3.0, 3.0, 100)

plt.figure(dpi=100, figsize=(8,6))
for activation in activations:
    activation_layer = layers.Activation(activation)
    y = activation_layer(x)
    plt.plot(x, y, label=activation)

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activations Functions')
plt.legend()
plt.grid(True)
plt.show()
