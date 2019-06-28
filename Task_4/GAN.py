import numpy as np
import pandas as pd

from tensorflow import keras
from keras.layers import Input, Dense, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding
from keras.models import Sequential, Model


class Data:
    """
    Class containing all training and testing data.
    """

    def __init__(self, filename, train=True):
        self.filename = filename
        self.train = train
        self.data = []
        self.features_train = []
        self.classes_train = []
        self.features_test = []

        """
        filename: str
            Path to data

        train: bool
            Whether or not data is used for training

        features_train: np.array
            Array of features used to train network

        classes_train: np.array
            Array of classes used to train network

        features_test: np.array
            Array of features to test the network on

        """

        self.readfile()
        self.segmentdata()

    def readfile(self):
        '''
        Reads h5 input files contaiing training and test data
        Input: N/A
        Output: data - numpy array [:, 121]
        '''
        if self.train:
            self.data = pd.read_hdf(self.filename, "train").values
        else:
            self.data = pd.read_hdf(self.filename, "test").values

    def segmentdata(self):
        '''
        Segments numpy arrays containing data into features and
        classes.
        Input: N/A
        Output: features_x - feature vectors within data for training,
                                validation, or testing
                classes_x  - class for given feature vector for training,
                                validation, or testing
        '''
        if self.train:
            self.features_train = self.data[:, 1:]
            self.classes_train = self.data[:, 0]
        else:
            self.features_test = self.data[:, :]


class ACGAN():
    def __init__(self, input_size, classes, epochs):
        # Input shape
        self.input_size = (1, input_size)
        self.num_classes = classes
        self.epochs = epochs
        self.latent_dim = 100
        self.batch_size = 128

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='sparse_categorical_crossentropy',
                                   optimizer='adam',
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        inp = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(inp)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam')

    def build_generator(self):

        model = Sequential()

        model.add(Dense(16, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(139, activation="relu"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        inp = model(model_input)

        return Model([noise, label], inp)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(1024, activation=keras.activations.relu, input_shape=self.input_size))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation=keras.activations.relu))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128, activation=keras.activations.relu))
        model.add(Dropout(0.5))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(64, activation=keras.activations.relu))
        model.add(Dropout(0.5))

        # model.summary()

        img = Input(shape=self.input_size)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        model.summary()

        return Model(img, [validity, label])

    def train(self, epochs, batch_size, sample_interval=50):

        self.batch_size = batch_size
        self.epochs = epochs

        # Load the dataset
        train = Data("train_labeled.h5", True)
        X_train = train.features_train
        Y_train = train.classes_train

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = np.expand_dims(X_train[idx], axis=0)
            imgs = np.swapaxes(imgs, 0, 1)

            print(imgs.shape)
            input("WAIT")

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (self.batch_size, 100))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 10, (self.batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9
            img_labels = Y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))


def Output(predictions, Id):
    '''
        Writes the predicted classes and their respective Id number
         into a csv file
        Input: predictions - predicted class
               Id          - case number for each prediction
        Output: N/A
    '''

    output = np.c_[np.arange(Id[0], Id[1]), predictions]

    with open('ouput.csv', "w") as file:
        file.write("Id,y\n")
        for row in range(output.shape[0]):
            file.write("{},{}\n".format(int(output[row, 0]),
                       int(output[row, 1])))


def main():
    acgan = ACGAN(139, 10, 100)
    acgan.train(epochs=100, batch_size=32, sample_interval=200)

    trainLabeled = Data("test.h5", True)
    trainUnlabeled = Data("test.h5", False)
    test = Data("test.h5", False)
    y_pred = np.argmax(acgan.discriminator.predict(test.features_test), axis=1)

    start = trainLabeled.features_train.shape[0] + trainUnlabeled.features_train.shape[0]
    end = start + test.features_test.shape[0]
    Id = [start, end]

    Output(y_pred, Id)


if __name__ == '__main__':
    main()
