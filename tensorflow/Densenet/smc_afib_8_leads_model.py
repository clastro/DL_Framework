import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from DBhandle import DBconn #DBhandle.py 파일 확인
from validation import show_metric
from FeatureEngineering.FeatureExtract.FeatureExtractionXML_DL import XMLWaveReader

db = DBconn('smc')
df_single_normal = db.selectTableData('single_normal_meta_data_all')
df_single_normal['label'] = 0

df_single_afib = db.selectTableData('single_afib_meta_data')
df_single_afib['label'] = 1

file_path = df_single_normal['Filepath'] + '/' + df_single_normal['Filename']

df = pd.concat([df_single_normal[['Filepath','Filename','label']],df_single_afib[['Filepath','Filename','label']]])
df.reset_index(drop=True,inplace=True)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, optimizers, metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Reshape, Input, Conv1D, BatchNormalization, ReLU, MaxPool1D, concatenate, AvgPool1D, GlobalAveragePooling1D
from tensorflow.keras import Sequential

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #초기화할 GPU Number
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#session.close()

def conv_layer(x, filters, kernel=1, strides=1):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel, strides=strides, padding="same")(x)
    return x

def dense_block(x, repetition, filters):
    for _ in range(repetition):
        y = conv_layer(x, 4 * filters)
        y = conv_layer(y, filters, 3)
        x = concatenate([y, x])
    return x

def transition_layer(x):
    x = conv_layer(x, x.shape[-1] // 2)
    x = AvgPool1D(2, strides=2, padding="same")(x)
    return x

def densenet(input_shape, n_classes, filters=32):

    input = Input(input_shape)
    x = Conv1D(64, 7, strides=2, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(3, strides=2, padding="same")(x)

    for repetition in [6, 12, 24, 16]:

        d = dense_block(x, repetition, filters)
        x = transition_layer(d)
    x = GlobalAveragePooling1D()(d)
    output = Dense(n_classes, activation="softmax")(x)

    model = Model(input, output)
    return model

input_shape = 8,5040
n_classes = 2

model = densenet(input_shape, n_classes, filters=32)
model.summary()

from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size=128, dim=(8, 5040), n_classes=2, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.label = df.label
        self.filepath = df['Filepath'] + '/' + df['Filename']
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(df.index))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filepath) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.filepath))

        indexes = self.indexes[start_idx:end_idx]
        list_file_temp = [self.filepath[k] for k in indexes]
        list_label_temp = [self.label[k] for k in indexes]

        X, y = self.__data_generation(list_file_temp, list_label_temp)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def normalize(self, data):
        mean_per_row = data.mean(axis = 1, keepdims=True)
        std_per_row = data.std(axis = 1, keepdims = True)
        return (data - mean_per_row ) / (std_per_row + 1e-10)

    def load_and_process_xml(self, xml_path):
        xmlread = XMLWaveReader(xml_path)
        return xmlread.get_data_array()

    def __data_generation(self, list_file_temp, list_label_temp):
        X = np.empty((len(list_file_temp), *self.dim))
        y = np.empty((len(list_label_temp)), dtype=int)

        indexes_to_remove = []

        for i, (filepath, label) in enumerate(zip(list_file_temp, list_label_temp)):
            ecg_wave = self.load_and_process_xml(filepath)

            if ecg_wave is not None:
                ecg_wave_padded = pad_sequences(ecg_wave, maxlen=5040, padding='post', truncating='post', dtype='float32')
                try:
                    X[i,] = ecg_wave_padded
                except ValueError as e:
                    indexes_to_remove.append(i)
                    print(e)
                    continue
                
            else:
                print(f"Skipping {filepath} because ecg_wave is None.")
                indexes_to_remove.append(i)
                continue

            y[i] = label

        if indexes_to_remove:
            X = np.delete(X, indexes_to_remove, axis=0)
            y = np.delete(y, indexes_to_remove, axis=0)

        return X, np.eye(self.n_classes)[y]

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

df_train = df.sample(n = 330000, random_state = 0)
df_valid = df.loc[~df.index.isin(df_train.index)]
df_train.reset_index(drop=True,inplace=True)
df_valid.reset_index(drop=True,inplace=True)

params = {'dim': (8,5040),
          'batch_size': 128,
          'n_classes': 2,
          'shuffle': False}

training_generator = DataGenerator(df_train, **params)
validation_generator = DataGenerator(df_valid, **params)

unique_labels, label_counts = np.unique(training_generator.label, return_counts=True)
class_weights = compute_class_weight(class_weight='balanced', classes = unique_labels, y = training_generator.label)
class_weight_dict = dict(zip(unique_labels, class_weights))

model.compile(loss=tf.keras.losses.binary_crossentropy, 
              optimizer='adam', 
              metrics=['accuracy',metrics.AUC()])

checkpoint_filepath = './dlruns/densenet/single_densenet_ver02_{epoch:02d}.h5'
# ver01 : BType Error Modificiation, 1 Lead -> 8 Leads, Normalization
# ver02 : No data Normalization 

# ModelCheckpoint 콜백 정의
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',  
    save_best_only=True,  
    save_weights_only=False,  # 모델 아키텍처와 가중치 모두 저장
    mode='min',  # 검증 손실을 최소화하는 방향으로 모델을 선택
    verbose=1
)

history = model.fit_generator(generator=training_generator,
                             validation_data=validation_generator, 
                             use_multiprocessing=True,
                             epochs=12,
                             verbose=1, 
                             class_weight=class_weight_dict,
                             callbacks=[checkpoint_callback])
