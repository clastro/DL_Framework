import tensorflow as tf

tensor = tf.random.uniform(shape=[28*28])
tensor_3d = tf.random.uniform(shape=[28*28*3])

# Reshape the grayscale image tensor into a vector
gray_vector = reshape(tensor, (28*28, 1))

# Reshape the color image tensor into a vector
color_vector = reshape(tensor_3d, (2352, 1))
