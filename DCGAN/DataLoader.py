import numpy as np
from keras.datasets import fashion_mnist
import random

class DataLoader:
    def __init__(self, noise_dim=100):
        self.dataset = []
        self.noise_dim = noise_dim

    def load_real(self):
        (train_x, _), (_, _) = fashion_mnist.load_data()
        self.dataset = np.expand_dims(train_x, axis=-1)

        self.dataset = self.dataset.astype('float32')
        self.dataset = (self.dataset - 127.5)/127.5
        np.random.shuffle(self.dataset)

    def next_real_batch(self, no_sample):
        indexes = np.random.randint(0, self.dataset.shape[0], no_sample)
        batch_x = self.dataset[indexes]
        batch_y = np.ones((no_sample, 1))

        return batch_x, batch_y

    def generate_noise(self, no_sample):
        batch_noise = np.random.randn(self.noise_dim*no_sample)
        batch_noise = batch_noise.reshape(no_sample, self.noise_dim)

        return batch_noise

    def next_fake_batch(self, generator, no_sample):
        batch_noise = self.generate_noise(no_sample)
        batch_x = generator.predict(batch_noise)

        batch_y = np.zeros((no_sample, 1))

        return batch_x, batch_y
