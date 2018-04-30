import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split

from spectrogram import Spectrogram
from utils import *

import json
import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=257, img_cols=251, channel=2):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.input_shape = img_cols * img_rows * channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G

        self.G = Sequential()
        dropout = 0.4
        encoding_depth = [64, 32, 16]
        decoding_depth = [32, 64]
        dim = self.input_shape
        
        for depth in encoding_depth + decoding_depth:
            self.G.add(Dense(depth, input_dim=dim))
            self.G.add(BatchNormalization(momentum=0.9))
            self.G.add(Activation('relu'))
            self.G.add(Dropout(dropout))
            dim = depth

        dim = decoding_depth[-1]
        self.G.add(Dense(self.input_shape, input_dim=dim))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Dropout(dropout))

        self.G.add(Reshape((self.img_rows, self.img_cols, self.channel)))

        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        # TODO[Siva]: define a loss function here
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self, params):
        self.params = params
        self.img_rows = 257
        self.img_cols = 251
        self.channel = 2

        self.get_data()

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def get_data(self):
        print ('Reading Data...')

        file_list_original = os.listdir(self.params['data_path'])
        file_list_original = [os.path.join(self.params['data_path'], i)
                for i in file_list_original[:]]

        mallet_file_list = [filename.replace("keyboard_", "mallet_", 2) 
                            for filename in file_list_original]

        mask = [i for i, filename in enumerate(mallet_file_list) 
                if os.path.isfile(filename)]

        file_list = [file_list_original[i] for i in mask]

        file_list = [filename for filename in file_list[50:60]]

        file_train, file_test = train_test_split(file_list, test_size=0.3)

        good_mallet_file_list = [filename.replace("keyboard_", "mallet_", 2) 
                                    for filename in file_train]
        
        spec_obj = Spectrogram(filenames = file_train)
        self.x_train = spec_obj.spectrogram
        self.x_train_img = spec_obj.images
        spec_obj.wav_to_spectrogram(good_mallet_file_list)
        self.y_train = np.concatenate([self.x_train, spec_obj.spectrogram],
                                        axis = -1)

        spec_obj.wav_to_spectrogram(file_test)
        self.x_test = spec_obj.spectrogram
        self.x_test_img = spec_obj.images
        good_mallet_file_list = [filename.replace("keyboard_", "mallet_", 2)
                                    for filename in file_test]
        spec_obj.wav_to_spectrogram(good_mallet_file_list)
        self.y_test = np.concatenate([self.x_test, spec_obj.spectrogram],
                                        axis = -1)

    def train(self, train_steps=2000, batch_size=256):
        for i in range(train_steps):
            batch_index = np.random.randint(0, self.x_train.shape[0],
                                            size=batch_size)
            images_train = self.x_train_img[batch_index, :, :, :]
            vec_train = self.x_train[batch_index, :]
            # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(vec_train)

            print "TRAIN SHAPE: ", images_train.shape
            print "TEST SHAPE: ", images_fake.shape

            x = np.concatenate((images_train, images_fake))
            print x.shape
            y = np.ones([2*batch_size, 1])
            print y.shape
            y[batch_size:, :] = 0
            print y[0]
            d_l oss = self.discriminator.train_on_batch(x, y)
            # d_loss = self.discriminator.fit(x, y,
            #                    epochs = 10,
            #                    batch_size = batch_size,
            #                    shuffle = True,
            #                    validation_data = (self.x_test, self.y_test),
            #                    callbacks=callbacks_list)

            y = np.ones([batch_size, 1])
            # noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            
            a_loss = self.adversarial.train_on_batch(vec_train, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)


if __name__ == '__main__':

    args = parse_arguments()
    with open(args.config) as f:
        params = json.load(f)

    create_folder(params['trained_weights_path'])

    agent = MNIST_DCGAN(params)

    if eval(params['train']):
        timer = ElapsedTimer()
        agent.train()
        agent.train(train_steps=10, batch_size=1)
        timer.elapsed_time()
    else:
        agent.test(params['test_data_path'])
