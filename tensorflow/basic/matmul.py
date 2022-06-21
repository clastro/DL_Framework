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
