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
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size>len(self.list_IDs)): #마지막 batch 데이터
            indexes = self.indexes[index*self.batch_size:len(self.list_IDs)]
        else:
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
        if(len(list_IDs_temp)<self.batch_size): #마지막 batch 내 데이터
            last_size = len(self.list_IDs) % self.batch_size
            X = np.empty((last_size, *self.dim))
            y = np.empty((last_size), dtype=int)
        else:
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

# Generators
training_generator = DataGenerator(df_part_train['unique_id'],df_part_train['label'], **params)
validation_generator = DataGenerator(df_test['unique_id'],df_test['label'], **params)
