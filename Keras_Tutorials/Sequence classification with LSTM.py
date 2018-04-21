#UseAI

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM

# =============================================================================
# # Generate Dummy Data
# import numpy as np
# x_train =  
# y_train = 
# x_test  = 
# y_test  =
# =============================================================================

# Build Model
model = Sequential()
model.add(Embedding(max_features, output_dim = 256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

# Training Configurations
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10, batch_size = 16)
score = model.evaluate(x_test, y_test, batch_size = 16)