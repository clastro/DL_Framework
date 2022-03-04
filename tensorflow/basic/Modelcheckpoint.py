from tensorflow.keras.callbacks import ModelCheckpoint #가장 높은 점수가 나온 모델의 파라미터를 저장할 때 쓰임

mc = ModelCheckpoint ('fashionMnistDNN.h5', monitor = 'val_acc', mode='max', verbose = 1, save_best_only=True)

model.fit(x_train,y_train, epochs=15, batch_size=128, validation_data = (x_valid, y_valid))
