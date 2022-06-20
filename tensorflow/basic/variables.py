import tensorflow as tf

a = tf.Variable([1,2,3,4,5,6], dtype=tf.float32)

a1 = a.numpy() # numpy로 변환

b = tf.constant(2, tf.float32)



