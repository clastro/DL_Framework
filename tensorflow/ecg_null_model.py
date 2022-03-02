import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import keras
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers,metrics
#!pip install tqdm
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
session.close()

df_train = pd.read_csv('df_train.csv',encoding='utf-8-sig')
df_test = pd.read_csv('df_test.csv',encoding='utf-8-sig')

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=128, dim=(5000,1), 
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp,list_labels_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp,list_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        
        #print(X.shape)

        # Generate data
        for i, (ID,label) in enumerate(zip(list_IDs_temp, list_labels_temp)):
            # Store sample:
            X[i,] = np.load('./smc_numpy_data/II/' + ID + '.npy').reshape(5000,1)
            y[i] = label 
            
        #for i, label in enumerate(list_labels_temp):      
            # Store class             

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
      
 params = {'dim': (5000,1),
          'batch_size': 128,
          'n_classes': 2,
          'shuffle': True}

# Datasets
#partition = # IDs
#labels = # Labels

# Generators
training_generator = DataGenerator(df_train['unique_id'],df_train['label'], **params)
validation_generator = DataGenerator(df_test['unique_id'],df_test['label'], **params)

# Design model
from tensorflow.keras import layers
model = Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(5000, 1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(loss=tf.keras.losses.binary_crossentropy, 
              optimizer='adam', 
              metrics=['accuracy',metrics.AUC()])
model.summary()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 5,
                    use_multiprocessing=True,
                    workers=2)
