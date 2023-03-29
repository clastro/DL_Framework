import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

file_h5 = './q_class_Best.h5'
loaded_model = tf.keras.models.load_model(file_h5)

ins = loaded_model.inputs
outs = loaded_model.layers[1].output
feature_map = tf.keras.Model(inputs=ins,outputs=outs)
feature_map.summary()

input_sig = np.expand_dims(sig_sample[230:1480],axis=1)
feature = feature_map.predict(input_sig)

plt.plot(feature[:,0:,1])
plt.show()
