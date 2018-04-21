from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential

# Dummy Data Generation
import numpy as np
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size = (1000, 1))
x_test  = np.random.random((100, 20))
y_test  = np.random.randint(2, size = (100, 1))

# Build Model
model = Sequential()
model.add(Dense(64, activation = 'relu', input_dim = 20))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

# Configure the training settings
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Train
model.fit(x_train, y_train, epochs = 20, batch_size = 128)
# Test the model 
score = model.evaluate(x_test, y_test, batch_size = 128)
