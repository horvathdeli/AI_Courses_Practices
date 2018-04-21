
from keras.layers import Input, LSTM, Dense, Embedding, concatenate
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers,
# between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.

main_input = Input(shape = (100,), dtype ='int32', name = 'main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(input_dim = 10000, input_length = 100, output_dim = 512)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

# Here we insert the auxiliary loss,
# allowing the LSTM and Embedding layer to be trained smoothly
# even though the main loss will be much higher in the model.

auxiliary_output = Dense(1, activation = 'sigmoid', name ='aux_output')(lstm_out)

# At this point, we feed into the model our auxiliary input data
# by concatenating it with the LSTM output:

auxiliary_input = Input(shape = (5,), name = 'aux_input')
x = concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation = 'sigmoid', name = 'main_output')(x)

# This defines a model with two inputs and two outputs:
model = Model(inputs = [main_input, auxiliary_input], outputs = [main_output, auxiliary_output])

# We compile the model and assign a weight of 0.2 to the auxiliary loss.
# To specify different loss_weights or  loss for each different output,
# you can use a list or a dictionary. Here we pass a single loss
# as the loss argument, so the same loss will be used on all outputs.
model.compile(optimizer ='rmsprop',
              loss = 'binary_crossentropy', 
              loss_weights = [1. ,0.2])

# We can train the model by passing it lists of input arrays
# and target arrays:
model.fit([headline_data, additional_data], [labels, labels],
          epochs = 50, batch_size = 32)

# =============================================================================
# Since our inputs and outputs are named (we passed them a "name" argument),
# we could also have compiled the model via:
# =============================================================================

model.compile(optimizer = 'rmsprop',
              loss = {'main_input': 'binary_crossentropy',
                      'aux_output': 'binary_crossentropy'},
              loss_weights = {'main_output': labels,
                              'aux_output':  labels})
