# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(12, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs and define a functional model
merged = keras.layers.add([m1_layer2, m2_layer2])
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)

# Print a model summary
print(model.summary())

'''
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 784)]        0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 784)]        0                                            
__________________________________________________________________________________________________
dense_16 (Dense)                (None, 12)           9420        input_1[0][0]                    
__________________________________________________________________________________________________
dense_18 (Dense)                (None, 12)           9420        input_2[0][0]                    
__________________________________________________________________________________________________
dense_17 (Dense)                (None, 4)            52          dense_16[0][0]                   
__________________________________________________________________________________________________
dense_19 (Dense)                (None, 4)            52          dense_18[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 4)            0           dense_17[0][0]                   
                                                                 dense_19[0][0]                   
==================================================================================================
Total params: 18,944
Trainable params: 18,944
Non-trainable params: 0
__________________________________________________________________________________________________

'''
