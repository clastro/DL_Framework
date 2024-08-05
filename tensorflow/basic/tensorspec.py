"""

tf.TensorSpec(
    shape,
    dtype=tf.dtypes.float32,
    name=None
)


TensorSpec is mostly used by tf.function to specify input signature.

"""

input_signature=(
    tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
    tf.TensorSpec(shape=[], dtype=tf.int32),
    tf.TensorSpec(shape=[], dtype=tf.float32),
)

