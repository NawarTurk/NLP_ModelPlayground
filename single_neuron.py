from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


# Create a network with one linear unit

#  Define a linear model¶
model = keras.Sequential([  # Sequential is used to build a model layer-by-layer in a linear stack
                            # works well for straightforward architectures
                            #  where each layer flows directly into the next without branching or merging.
    layers.Dense(  #  creates a fully connected layer 
                   # where every neuron in the layer is connected to every neuron in the previous layer.
        units=1,  # units=1 specifies that this layer will have one neuron,
        input_shape=[1]   # input_shape=[3] specifies the input has 3 features (input dimension). 
                          # Setting input_shape=[3] would create a network accepting vectors of length 3, like [0.2, 0.4, 0.6].)
    )
])

#  Look at the weights¶
w, b = model.get_weights()

x = tf.linspace(-1.0, 1.0, 10)  # produces a 1D tensor of 10 evenly spaced values ranging from -1 to 1, inclusive. (10,)
y = model.predict(x)

print(y.shape)
print(y)
print(b)

                 

