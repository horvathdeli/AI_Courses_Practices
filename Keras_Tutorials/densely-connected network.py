from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape = (784,))

# a layer instance is a callable on a tensor, and returns a tensor
x = Dense(64, activation = 'relu')(inputs)
x = Dense(64, activation = 'relu')(x)
predictions = Dense (10, activation = 'softmax')(x)

# This creates a model that includes 
# the input layer and three Dense layers
model = Model(inputs = inputs, outputs = predictions)
model.compile(optimizer = 'rmsprop',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
#Start Training
model.fit(data, labels) 

x = Input(shape(784,))

# This works and returns the 10-way softmay as we defined above
y = model(x)

from keras.layers import TimeDistributed

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape = (20, 784))

# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)