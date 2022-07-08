from tensorflow.keras.utils import Sequence

class Dataloader(Sequence):

    def __init__(self, x_set, y_set,n_classes, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_id = [self.x[i] for i in indices]
        batch_x = [*map(self.load_numpy,batch_id)]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), tf.keras.utils.to_categorical(batch_y, num_classes=self.n_classes) #,np.array(batch_y)
    
    def load_numpy(self,unique_id):
        ecg_wave = np.load('/smc_work/datanvme/smc/origin/' + unique_id + '.npy')[10:4990,:12]
        return ecg_wave

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
