import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
# from keras.layers import Dense, Dropout


class Data:
    def __init__(self, filename, train=True):
        self.filename = filename
        self.train = train
        self.data = []
        self.features = []
        self.classes = []

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
        Output: features - feature vectors within data
                classes  - class for given feature vector
        '''

        if self.train:
            self.features = self.data[:, 1:]
            self.classes = self.data[:, 0]
        else:
            self.features = self.data[:, :]
            self.classes = []


class NeuralNet:
    def __init__(self, train, epochs, k_fold=10):
        self.train = train
        self.k_fold = k_fold
        self.epochs = epochs
        self.model = []

        self.trainnet()

    def trainnet(self):
        tf.set_random_seed(100)
        self.model = keras.Sequential([
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(5, activation='softmax')])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(self.train.features, self.train.classes,
                       epochs=self.epochs)


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

    neuralnet = NeuralNet(train, 100)
    y_pred = np.argmax(neuralnet.model.predict(test.features), axis=1)

    start = train.features.shape[0]
    end = start + test.features.shape[0]
    Id = [start, end]

    Output(y_pred, Id)


if __name__ == '__main__':
    main()
