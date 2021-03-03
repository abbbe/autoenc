# https://blog.keras.io/building-autoencoders-in-keras.html
# "Convolutional autoencoder" + bottleneck

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def build_encoder(input_img, latentDim, build_params):
    x = input_img

    #print("before enc conv=" + str(x))
    for conv in build_params['convs']:
        x = layers.Conv2D(conv, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

#    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
#    x = layers.MaxPooling2D((2, 2), padding='same')(x)
#    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#    x = layers.MaxPooling2D((2, 2), padding='same')(x)
#    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    #print("after enc conv=" + str(x))

    volumeSize = K.int_shape(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(latentDim)(x)

    z_mean = layers.Dense(latentDim, name="z_mean")(x)
    z_log_var = layers.Dense(latentDim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(input_img, [z_mean, z_log_var, z], name="encoder")
    
    return encoder, volumeSize

def build_decoder(encoded, volumeSize, build_params):
    x = encoded

    x = layers.Dense(np.prod(volumeSize[1:]))(encoded)
    x = layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    #print("before dec conv=" + str(x))
    rev_convs = list(build_params['convs'])
    rev_convs.reverse()
    for conv in rev_convs:
        x = layers.Conv2D(conv, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
    #x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = layers.UpSampling2D((2, 2))(x)
    #x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = layers.UpSampling2D((2, 2))(x)
    #x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    #x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    #print("after dec conv=" + str(x))

    decoded = x
    decoder = keras.Model(encoded, decoded, name="decoder")
    return decoder

def build_autoencoder(inputShape, latentShape, build_params):
    inputs = layers.Input(shape=inputShape)
    encoder_model, volumeSize = build_encoder(inputs, latentShape, build_params)

    latent = layers.Input(shape=latentShape)
    decoder_model = build_decoder(latent, volumeSize, build_params)

    autoencoder_model = VAE(encoder_model, decoder_model, name="autoencoder")
    autoencoder_model.compile(optimizer=keras.optimizers.Adam())

    return {'ae': autoencoder_model, 'enc': encoder_model, 'dec': decoder_model}

