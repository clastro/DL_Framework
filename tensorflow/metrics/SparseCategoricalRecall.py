class SparseCategoricalRecall(tf.keras.metrics.Metric):
    def __init__(self, name='sparse_categorical_recall', **kwargs):
        super(SparseCategoricalRecall, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        for i in range(tf.reduce_max(y_true) + 1):
            true_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
            false_neg = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.not_equal(y_pred, i)), tf.float32))
            self.true_positives.assign_add(true_pos)
            self.false_negatives.assign_add(false_neg)

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)
