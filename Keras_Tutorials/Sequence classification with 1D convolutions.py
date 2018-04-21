from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

# =============================================================================
# # Generate Dummy data
# 
# =============================================================================

# Arbitrary
seq_length = None

# Build the model
model = Sequential()
model.add(Conv1D(64, 3, activation = 'relu', input_shape = (seq_length, 100)))
model.add(Conv1D(64, 3, activation = 'relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation = 'relu'))
model.add(Conv1D(128, 3, activation = 'relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

#Configuration for Training
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
#Train
model.fit(x_train, y_train, epochs = 10, batch_size = 16)
#Test
model.evaluate(x_test, y_test, batch_size = 16)