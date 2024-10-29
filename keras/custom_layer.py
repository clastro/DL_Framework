class CustomOperationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        
        standardized = (inputs - mean) / (std + 1e-7)  # 작은 값으로 나눔, 나눗셈에서 0 방지
        
        return standardized

model = keras.models.load_model('./model/model.h5', custom_objects={'CustomOperationLayer': CustomOperationLayer})
