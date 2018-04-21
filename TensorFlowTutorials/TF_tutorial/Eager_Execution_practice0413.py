# Eager Execution ~ TensorFlow
import tensorflow as tf

tf.enable_eager_execution()

tf.executing_eagerly()

x = [[2.]]

m = tf.matmul(x, x)
print("hello, {}".format(m))
print("\n")
print(m)

import tensorflow.contrib.eager as tfe

w = tfe.Variable([[1.0]])
with tfe.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, [w])

print("\n {} \n".format(grad))

NUM_E = 1000
training_inputs = tf.random_normal([NUM_E])
noise = tf.random_normal([NUM_E])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
    return input * weight + bias

def loss(weights, biases):
    error = prediction(
            training_inputs,
            weights,
            biases) - training_outputs
    return tf.reduce_mean(tf.square(error))

def grad(weights, biases):
    with tfe.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value,
                         [weights, biases])

train_steps = 200
learning_rate = 0.01

W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial Loss: {:.3f}".format(loss(W,B)))

for i in range(train_steps):
    dW, dB = grad(W, B)
    W.assign_sub(dW * learning_rate)
    B.assign_sub(dB * learning_rate)
    if i % 20 == 0:
        print("Loss at step: {:03d} : {:.3f}".format(i, loss(W, B)))

print("Loss at the end: {:.3f}".format(loss(W, B)))
#import numpy as np
print(" Weight: {:.5f} \n Bias: {:.5f}".format(W.numpy(), B.numpy()))
        
# =============================================================================
# Replay the tfe.GradientTape to compute the gradients
# and apply them in a training loop.
# =============================================================================

dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                      data.train.labels))
print(dataset)   
    
    
    
    
    