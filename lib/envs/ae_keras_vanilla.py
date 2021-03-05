# https://blog.keras.io/building-autoencoders-in-keras.html
# "Convolutional autoencoder" + bottleneck

import numpy as np

from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.backend as K

def build_encoder(input_img, latentDim):
    x = input_img

    #print("before enc conv=" + str(x))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    #print("after enc conv=" + str(x))

    volumeSize = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latentDim)(x)

    encoded = x
    encoder = models.Model(input_img, encoded, name="encoder")
    return encoder, volumeSize

def build_decoder(encoded, volumeSize):
    x = encoded

    x = layers.Dense(np.prod(volumeSize[1:]))(encoded)
    x = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    #print("before dec conv=" + str(x))
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    #print("after dec conv=" + str(x))

    decoded = x
    decoder = models.Model(encoded, decoded, name="decoder")
    return decoder

def build_conv_autoencoder(inputShape, latentShape):
    inputs = layers.Input(shape=inputShape)
    encoder_model, volumeSize = build_encoder(inputs, latentShape)

    latent = layers.Input(shape=latentShape)
    decoder_model = build_decoder(latent, volumeSize)

    autoencoder_model = models.Model(inputs, decoder_model(encoder_model(inputs)), name="autoencoder")
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return {'ae': autoencoder_model, 'enc': encoder_model, 'dec': decoder_model}
