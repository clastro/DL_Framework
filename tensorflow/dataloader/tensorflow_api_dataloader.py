import numpy as np
import tensorflow as tf

def create_dataloader(X, y, batch_size=64, shuffle=True, num_classes=None, seq_length=None):
    """
    Create a TensorFlow data loader with batching and optional shuffling.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Labels.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_classes (int): Number of classes for one-hot encoding.
        seq_length (int): Length of the sequence (e.g., 561).

    Returns:
        tf.data.Dataset: TensorFlow dataset ready for training or evaluation.
    """
    # Convert numpy arrays to TensorFlow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    # Ensure labels have the shape [batch_size, seq_length] for sequence classification
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle dataset if specified
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance improvement
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


