import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras


def ReadFile(filename):
    data = pd.read_csv(filename)
    return data


def main():
    tests = 5
    train_data = ReadFile('train.csv')
    test_data = ReadFile('test.csv')
    n = test_data.shape[0]

    x = train_data.iloc[:, 2:].values
    y = train_data.iloc[:, 1].values
    x_test = test_data.iloc[:, 1:].values
    y_pred = np.zeros([n, tests])

    for test in range(tests):
        tf.set_random_seed(test)
        model = keras.Sequential([
            keras.layers.Dense(256, activation=tf.nn.elu),
            keras.layers.Dense(50, activation=tf.nn.elu),
            keras.layers.Dense(256, activation=tf.nn.elu),
            keras.layers.Dense(3, activation=tf.nn.softmax)])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        model.fit(x, y, epochs=100)

        y_pred[:, test] = np.argmax(model.predict(x_test), axis=1)

    u, indices = np.unique(y_pred, return_inverse=True)
    output = u[np.argmax(np.apply_along_axis(np.bincount, 1,
               indices.reshape(y_pred.shape), None,
               np.max(indices) + 1), axis=1)]

    ouput = np.c_[np.arange(n) + 2000, y_pred]

    with open('ouput.csv', "w") as file:
        file.write("Id,y\n")
        for row in range(output.shape[0]):
            file.write("{},{}\n".format(int(ouput[row, 0]),
                       int(ouput[row, 1])))


if __name__ == '__main__':
    main()
