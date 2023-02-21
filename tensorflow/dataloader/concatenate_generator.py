import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import concatenate
from scipy.interpolate import interp1d
import numpy as np
import pickle

def get_model():
    input_tens = tf.keras.Input(shape=(125*2*5,1))
    input_tens_diff = tf.keras.Input(shape=(125*2*5 - 1 ,1))

    x = tf.keras.layers.Conv1D(32, kernel_size=30, padding="same", activation="relu")(input_tens)
    x = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=5)(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    
    x = tf.keras.layers.LSTM(16)(x)
    
    x = tf.keras.Model(inputs=input_tens, outputs=x)
    
    y = tf.keras.layers.Conv1D(32, kernel_size=30, padding="same", activation="relu")(input_tens_diff)
    y = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool1D(pool_size=5)(y)

    y = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool1D(pool_size=2)(y)

    y = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(y)
    y = tf.keras.layers.Conv1D(32, kernel_size=10, padding="same", activation="relu")(y)
    y = tf.keras.layers.MaxPool1D(pool_size=2)(y)
    
    y = tf.keras.layers.LSTM(16)(y)
    y = tf.keras.Model(inputs=input_tens_diff, outputs=y)
    
    combined = concatenate([x.output, y.output])
    
    z = tf.keras.layers.Dense(2, activation="softmax")(combined)
    model = tf.keras.Model(inputs=[input_tens,input_tens_diff], outputs=z)

    print(model.summary())
    return model


class DataloaderNormUnderSampling(Sequence):
    def __init__(self, id_list, ann_list, batch_size, shuffle=False,diff=False):
        self.id_list = id_list
        self.ann_list = ann_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.diff = diff
        self.total_number_of_one_epoch = 0   # For undersampling, Total number of data in one epoch need to be changed
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.total_number_of_one_epoch / self.batch_size))

    def on_epoch_end(self):
        # Get Index for all
        all_idx = np.arange(len(self.id_list))
        # Index for normal(ann==0)
        normal_idx = all_idx[self.ann_list==0]
        # Index for not normal(ann!=0)
        notnormal_idx = all_idx[self.ann_list!=0]
        # Shuffle and get 10% of all
        np.random.shuffle(normal_idx)
        normal_idx_undersampled = normal_idx[:int(len(normal_idx)*0.1)]
        # Concat & Shuffle
        self.indices = np.concatenate((normal_idx_undersampled, notnormal_idx), axis=0)
        self.total_number_of_one_epoch = len(self.indices)
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        inds_ = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        id_list_tmp = [self.id_list[i] for i in inds_]
        x_,diff_x_, y_ = self.__data_generation(id_list_tmp)
        # Change shape for model input
        return [x_.reshape(self.batch_size, -1, 1),diff_x_.reshape(self.batch_size, -1, 1)], y_.reshape(self.batch_size, 1)

    def __data_generation(self, id_list_tmp):
      
        max_size = 125*2*5   # For fixed size
        diff_max_size = max_size - 1
        
        x_ = np.zeros((self.batch_size, max_size), dtype=np.float32)
        diff_x_ = np.zeros((self.batch_size, diff_max_size), dtype=np.float32)
        y_ = np.zeros((self.batch_size), dtype=np.int32)
        
        for idx, val in enumerate(id_list_tmp):
            loaded_data = np.load(f"/wellysis/datanvme/BeatClass/230117/beats5/{val}.npy")
            diff_loaded_data = np.diff(loaded_data)
            
            input_beat_length = len(loaded_data)
            diff_input_beat_length = len(diff_loaded_data)
            
            x = np.linspace(0, input_beat_length, num=input_beat_length, endpoint=True)
            diff_x = np.linspace(0, diff_input_beat_length, num=diff_input_beat_length, endpoint=True)
            
            interpolated_beat_length = int(max_size)
            interpolated_beat = np.linspace(0, input_beat_length, num=interpolated_beat_length, endpoint=True)
            
            diff_interpolated_beat_length = int(diff_max_size)
            diff_interpolated_beat = np.linspace(0, diff_input_beat_length, num=diff_interpolated_beat_length, endpoint=True)
            
            x_[idx,:] = interpolated_beat
            diff_x_[idx,:] = diff_interpolated_beat
            ann_val = int(val.split("_")[-1])
            if ann_val == 4:
                ann_val_input = 1
            else:
                ann_val_input = 0

            y_[idx] = ann_val_input  # Last value (splited by "_") of filename is label

        return [x_,diff_x_], y_

        
if __name__ == "__main__":
    train_ids = np.load("/train.npy")
    val_ids = np.load("/valid.npy")
    train_label = np.array([int(id_.split("_")[-1]) for id_ in train_ids])
    val_label = np.array([int(id_.split("_")[-1]) for id_ in val_ids])
    
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    
    train_dataloader = DataloaderNormUnderSampling(train_ids, train_label, 256, True)
    val_dataloader = DataloaderNormUnderSampling(val_ids, val_label, 256, False)

    model.fit(train_dataloader, epochs=300, validation_data=val_dataloader)

    model.save('/s_class')

