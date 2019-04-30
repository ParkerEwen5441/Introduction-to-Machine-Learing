import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


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


class NeuralNet:
    def __init__(self, train, epochs):
        self.train = train
        self.epochs = epochs
        self.model = []

        self.trainnet()

    def trainnet(self):
        tf.set_random_seed(100)
        self.model = keras.Sequential([
            keras.layers.Dense(1024, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(5, activation='softmax')])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        # es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001)
        # self.model.fit(self.train.features_train, self.train.classes_train,
        #                batch_size=900, epochs=self.epochs,
        #                callbacks=[es], validation_split=0.1)
        self.model.fit(self.train.features_train, self.train.classes_train,
                       batch_size=900, epochs=self.epochs,
                       validation_split=0.1)


def PCAApply(train, test):
    scaler = StandardScaler()
    scaler.fit(train.features_train)
    train.features_train = scaler.transform(train.features_train)
    test.features_test = scaler.transform(test.features_test)

    pca = PCA(.99)
    pca.fit(train.features_train)
    train.features_train = pca.transform(train.features_train)
    test.features_test = pca.transform(test.features_test)


def Output(predictions, Id):
    output = np.c_[np.arange(Id[0], Id[1]), predictions]

    with open('ouput.csv', "w") as file:
        file.write("Id,y\n")
        for row in range(output.shape[0]):
            file.write("{},{}\n".format(int(output[row, 0]),
                       int(output[row, 1])))


def main():
    train = Data("train.h5", True)
    test = Data("test.h5", False)

    PCAApply(train, test)

    neuralnet = NeuralNet(train, 150)
    y_pred = np.argmax(neuralnet.model.predict(test.features_test), axis=1)

    start = train.features_train.shape[0]
    end = start + test.features_test.shape[0]
    Id = [start, end]

    Output(y_pred, Id)


if __name__ == '__main__':
    main()
