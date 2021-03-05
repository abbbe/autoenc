import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import numpy as np

def build_encoder_layer(x, latentDim):
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    volumeSize = K.int_shape(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latentDim)(x)
    return latent, volumeSize

def build_decoder_layer(latent, volumeSize):
    x = layers.Dense(np.prod(volumeSize[1:]))(latent)
    x = layers.Reshape(volumeSize[1:])(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

def build_conv_autoencoder(inputShape, latentDim):   
    inputs = layers.Input(shape=inputShape)

    encoderL, encoderLvolumeSize = build_encoder_layer(inputs, latentDim=1)
    encoderM, encoderMvolumeSize = build_encoder_layer(inputs, latentDim=1)
    encoder_model = models.Model(inputs, [encoderL, encoderM], name="encoder")

    latentL, latentM = layers.Input(shape=(1,)), layers.Input(shape=(1,))
    decoderL = build_decoder_layer(latentL, encoderLvolumeSize)
    decoderM = build_decoder_layer(latentM, encoderMvolumeSize)

    decoderL_flat = layers.Flatten()(decoderL)
    decoderM_flat = layers.Flatten()(decoderM)
    decoder_LM_flat = tf.keras.layers.Add()([decoderL_flat, decoderM_flat])
    decoder_LM = layers.Reshape(inputShape)(decoder_LM_flat)
    latent = [latentL, latentM]
    decoder_model = models.Model(latent, decoder_LM, name='decoder')

    autoencoder = models.Model(inputs, decoder_model(encoder_model(inputs)), name="autoencoder")
    #autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    opt = optimizers.Adam(learning_rate=0.005)
    autoencoder.compile(optimizer=opt, loss='mse')
    
    return {'ae': autoencoder, 'enc': encoder_model, 'dec': decoder_model }
