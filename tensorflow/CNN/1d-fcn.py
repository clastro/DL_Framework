def block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
    '''
    Defines a block in the VGG block
 
    Args:
        x(tensor) -- input image
        n_convs(int) -- number of convolution lyaers to append
        filters(int) -- number of filters for the convolution lyaers
        activation(string or object) -- activation to use in the convolution
        pool_size(int) -- size of the pooling layer
        pool_stride(int) -- stride of the pooling layer
        block_name(string) -- name of the block
    
    Returns:
        tensor containing the max-pooled output of the convolutions
    '''
    for i in range(n_convs):
        x = tf.keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   activation=activation,
                                   padding='same',
                                   name=f'{block_name}_conv{i+1}')(x)
    
    x = tf.keras.layers.MaxPooling1D(pool_size=pool_size,
                                     strides=pool_stride,
                                     name=f'{block_name}_pool{i+1}')(x)
    
    return x

def VGG_16(image_input):
    '''
    This function defines the VGG encoder.
 
    Args:
        image_input(tensor) -- batch of images
    
    Returns:
        tuple of tensors -- output of all encoder blocks plus the final convolution layer
    '''
 
    # create 5 blocks with increasing filters at each stage
    x = block(image_input, n_convs=2, filters=64, kernel_size=3, activation='relu',
              pool_size=2, pool_stride=2,
              block_name='block1')
    p1 = x # (input/2, 64)
 
    x = block(x, n_convs=2, filters=128, kernel_size=3, activation='relu',
              pool_size=2, pool_stride=2,
              block_name='block2')
    p2 = x # (input/4, 128)
 
    x = block(x, n_convs=3, filters=256, kernel_size=3, activation='relu',
              pool_size=2, pool_stride=2,
              block_name='block3')
    p3 = x # (input/8, 256)
 
    x = block(x, n_convs=3, filters=512, kernel_size=3, activation='relu',
              pool_size=2, pool_stride=2,
              block_name='block4')
    p4 = x # (input/16, 512)
 
    x = block(x, n_convs=3, filters=512, kernel_size=3, activation='relu',
              pool_size=2, pool_stride=2,
              block_name='block5')
    p5 = x # (input/32, 512)
 
    # create the vgg model
    vgg = tf.keras.Model(image_input, p5)
    # load the pretrained weights downloaded
    #vgg.load_weights(vgg_weights_path)
    # number of filters for the output convolutional layers
    n = 4096
    # our input signals are 5000 datapoints so they will be downsampled to 7 after the pooling layers above.
    # we can extract more features by chaining two more convolution layers.
    c6 = tf.keras.layers.Conv1D( n , 7 , activation='relu' , padding='same', name="conv6")(p5)
    c7 = tf.keras.layers.Conv1D( n , 1 , activation='relu' , padding='same', name="conv7")(c6)
 
    # return the outputs at each stage. you will only need two of these in this particular exercise 
    # but we included it all in case you want to experiment with other types of decoders.
    return (p1, p2, p3, p4, c7)


def decoder(convs, n_classes):
    '''
    Defines the FCN 32,16,8 decoder.
 
    Args:
        convs(tuple of tensors) -- output of the encoder network
        n_classes(int) -- number of classes
    
    Returns:
        tensor with shape (height, width, n_classes) contating class probabilities(FCN-32, FCN-16, FCN-8)
    '''
 
    # unpack the output of the encoder
    f1, f2, f3, f4, f5 = convs 
 
    # FCN-32 output
    fcn32_o = tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=32, strides=32, use_bias=False)(f5)
    fcn32_o = tf.keras.layers.Activation('softmax')(fcn32_o)
 
    # upsample the output of the encoder then crop extra pixels that were introduced
    o = tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=4, strides=2, use_bias=False)(f5)
    o = tf.keras.layers.Cropping1D(cropping=1)(o) 
 
    # load the pool4 prediction and do a 1x1 convolution to reshape it to the same shape of 'o' above
    o2 = f4 
    o2 = tf.keras.layers.Conv1D(n_classes, 1, activation='relu', padding='same')(o2) 
 
    # add the result of the upsampling and pool4 prediction
    o = tf.keras.layers.Add()([o, o2]) 
 
    # FCN-16 output
    fcn16_o = tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=16, strides=16, use_bias=False)(o)
    fcn16_o = tf.keras.layers.Activation('softmax')(fcn16_o)
 
    # upsample the resulting tensor of the operation you just did
    o = tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=4, strides=2, use_bias=False)(o)
    o = tf.keras.layers.Cropping1D(cropping=1)(o) 
 
    # load the pool3 prediction and do a 1x1 convolution to reshape it to shame shape of 'o' above
    o2 = f3 
    o2 = tf.keras.layers.Conv1D(n_classes, 1, activation='relu', padding='same')(o2)
 
    # add the result of the upsampling and pool3 prediction
    o = tf.keras.layers.Add()([o, o2])
 
    # upsample up to the size of the original signal
    o = tf.keras.layers.Conv1DTranspose(n_classes, kernel_size=8, strides=8, use_bias=False)(o) 
 
    # append a softmax to get the class probabilities
    fcn8_o = tf.keras.layers.Activation('softmax')(o)
 
    return fcn32_o, fcn16_o, fcn8_o

def segmentation_model():
    '''
    Defines the final segmentation model by chaining together the encoder and decoder.
 
    Returns:
        Keras Model that connects the encoder and decoder networks of the segmentation model
    '''
 
    inputs = tf.keras.layers.Input(shape=(5120,1))
    convs = VGG_16(inputs)
    fcn32, fcn16, fcn8 = decoder(convs, 2)
    model_fcn32 = tf.keras.Model(inputs, fcn32)
    model_fcn16 = tf.keras.Model(inputs, fcn16)
    model_fcn8 = tf.keras.Model(inputs, fcn8)
 
    return model_fcn32, model_fcn16, model_fcn8
