from tensorflow import constant
from tensorflow import Variable

# Define features, params, and bill as constants

features = Variable([[2, 24], [2, 26], [2, 57], [1, 37]])
params = constant([[1000], [150]])
bill = Variable([[3913], [2682], [8617], [64400]])

# Compute billpred using features and params
billpred = matmul(features,params)

# Compute and print the error
error = bill - billpred
print(error.numpy())

import tensorflow as tf
# Reshape model from a 1x3 to a 3x1 tensor

model = tf.reshape(model,(3, 1))
# Multiply letter by model
output = tf.matmul(letter, model)
# Sum over output and print prediction using the numpy method
prediction = tf.reduce_sum(output)

print(prediction.numpy())
