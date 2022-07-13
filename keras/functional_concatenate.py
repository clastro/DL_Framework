import numpy as np 
import pandas as pd
import re

from tensorflow.keras import Input,initializers, regularizers, constraints, optimizers
from tensorflow.keras.layers import LSTM , Embedding, Dropout , Activation, GRU, Flatten, Bidirectional, GlobalMaxPool1D, Convolution1D, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow import keras

input_cause = Input(shape=(None,),name='cause')
x = Embedding(max_features, embed_size)(input_cause)
x = Bidirectional(LSTM(32, return_sequences = True))(x)
x = BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = Dropout(0.05)(x)
x = GlobalMaxPool1D()(x)

input_attitude = Input(shape=(None,),name='attitude')
y = Embedding(max_features, embed_size)(input_attitude)
y = Bidirectional(LSTM(32, return_sequences = True))(y)
y = BatchNormalization()(y)
y = Dense(20, activation='relu')(y)
y = Dropout(0.05)(y)
y = GlobalMaxPool1D()(y)

input_function = Input(shape=(None,),name='function')
z = Embedding(max_features, embed_size)(input_function)
z = Bidirectional(LSTM(32, return_sequences = True))(z)
z = BatchNormalization()(z)
z = Dense(20, activation='relu')(z)
z = Dropout(0.05)(z)
z = GlobalMaxPool1D()(z)

w = concatenate([x, y, z])
w = Dense(20, activation='relu')(w)
out =  Dense(1, activation='sigmoid')(w)

model = Model(inputs=[input_cause, input_attitude, input_function], outputs=out)
model.compile(loss='binary_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])

#model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
model_check = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

batch_size = 128 
epochs = 20
history = model.fit([X_cause_train,X_attitude_train,X_function_train],y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,callbacks=[es,model_check])

pred = model.predict([X_cause_test,X_attitude_test,X_function_test])


