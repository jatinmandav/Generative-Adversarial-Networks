from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Dropout, BatchNormalization, Activation, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


class DCGAN:
    def __init__(self, img_rows=28, img_cols=28, channels=1, noise_dim=100):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.noise_dim = noise_dim

    def build_generator(self):
        model = Sequential()

        model.add(Dense(7*7*128, activation='relu', input_dim=self.noise_dim))
        model.add(Reshape((7, 7, 128)))

        model.add(Conv2DTranspose(128, kernel_size=3, strides=(2, 2), padding='same'))
        model.add(Conv2D(128, kernel_size=3, kernel_initializer='glorot_uniform', padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(64, kernel_size=3, strides=(2, 2), padding='same'))
        model.add(Conv2D(128, kernel_size=3, kernel_initializer='glorot_uniform', padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2D(self.channels, kernel_size=3, kernel_initializer='glorot_uniform', padding='same'))
        model.add(Activation('tanh'))

        noise_input = Input(shape=(self.noise_dim,))
        img = model(noise_input)
        return Model(noise_input, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_gan(self):
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        noise = Input(shape=(self.noise_dim,))
        img = self.generator(noise)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.dcgan = Model(noise, valid)
        self.dcgan.compile(loss='binary_crossentropy', optimizer=optimizer)
