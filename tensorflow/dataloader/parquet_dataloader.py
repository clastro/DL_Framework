import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf

class ECGDataLoader:
    def __init__(self, uid_list, batch_size=128, chunk_size=3840, label_mapping=None, num_classes = 2):
        self.uid_list = uid_list
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.label_mapping = label_mapping if label_mapping is not None else {
            "Q": 1,
            "N": 0,
            "S": 0,
            "V": 0,
            "": 1
        }
        self.num_classes = num_classes

    # Function to load a single UID's data
    def load_parquet_data(self, uid):
        parquet_file = f'/wellysis/datanvme/BeatClass/krprod_{uid}.parquet'
        read_table = pq.read_table(parquet_file)
        df = read_table.to_pandas()
        return df
    
    # Function to map labels based on the predefined label_mapping
    def map_labels(self, labels):
        # Apply label mapping (e.g., 'Q' -> 1, 'N' -> 0)
        vectorized_map = np.vectorize(lambda x: self.label_mapping.get(x, 0))
        mapped_labels = vectorized_map(labels)
        # Convert to one-hot encoding
        one_hot_labels = np.eye(self.num_classes)[mapped_labels]
        
        return one_hot_labels
    

    # Function to create labels based on idx_beat and ann_beat
    def create_labels(self, idx_beat, ann_beat):
        size = idx_beat[-1] + 1  # Assuming the last beat index represents the size of the signal
        labels = np.zeros(size, dtype=object)

        start_idx = 0

        for i in range(len(idx_beat)):
            end_idx = idx_beat[i]
            label = ann_beat[i]
            labels[start_idx:end_idx + 1] = label
            start_idx = end_idx + 1
            
        return self.map_labels(labels)

    # Function to split data into chunks of chunk_size
    def split_data(self, data, chunk_size):        
        total_length = len(data)
        num_chunks = total_length // chunk_size
        return np.array_split(data[:num_chunks * chunk_size], num_chunks)
    # Generator function for data loading and processing
    
                
    def data_generator(self):
        for uid in self.uid_list:
            df = self.load_parquet_data(uid)

            # Extract relevant columns
            signals = df['signal'][0]  # Signal data
            idx_beat = df['idx_beat'][0]  # Beat indices
            ann_beat = df['ann_beat'][0]  # Beat annotations

            # Create labels for the current UID
            labels = self.create_labels(idx_beat, ann_beat)
            
            if len(signals) != len(labels):
                min_length = min(len(signals), len(labels))
                signals = signals[:min_length]
                labels = labels[:min_length]

            # Split signals and labels into chunks
            signals_chunks = self.split_data(signals, self.chunk_size)
            labels_chunks = self.split_data(labels, self.chunk_size)

            # Yield each batch of signals and labels
            for signal_chunk, label_chunk in zip(signals_chunks, labels_chunks):
                # Apply label mapping
                #label_chunk = self.map_labels(label_chunk)
                yield signal_chunk, label_chunk
                
    def create_tf_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator, 
            output_signature=(
                tf.TensorSpec(shape=(self.chunk_size, ), dtype=tf.float32),  # assuming signal is float32
                tf.TensorSpec(shape=(self.chunk_size, self.num_classes), dtype=tf.float32)  # one-hot encoded label
            )
        )

        # Shuffle and batch the dataset
        dataset = dataset.shuffle(buffer_size=100000).batch(self.batch_size)

        return dataset
