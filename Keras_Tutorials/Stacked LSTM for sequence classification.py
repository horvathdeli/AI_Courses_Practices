from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np 

data_dim = 16 
timesteps = 8
num_classes = 10

# Generate Dummy Training Data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate Dummy Validation Data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

# =============================================================================
# # Generate Dummy Validation Data
# x_test = np.random.random((100, timesteps, data_dim))
# y_test = np.random.random((100, num_classes))
# 
# =============================================================================
#Build the Model
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
# returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences = True,
               input_shape = (timesteps, data_dim))) 
# returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences = True))
# return a single vector of dimension 32
model.add(LSTM(32))
model.add(Dense(10, activation = 'softmax'))

#Training Configurations
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Training 
model.fit(x_train, y_train, epochs = 10, batch_size = 64,
          validation_data = (x_val, y_val))
# =============================================================================
# # Test
# score = model.evaluate(x_test, y_test, batch_size = 64)
# print(score)
# =============================================================================
