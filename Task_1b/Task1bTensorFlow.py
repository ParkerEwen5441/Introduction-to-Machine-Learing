import numpy as np
import pandas as pd
import tensorflow as tf

import csv


def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size,
                 chunk_size)


def ReadFile():
    data = pd.read_csv('train.csv')
    return data


def x2Phi(data):
    data = data.values

    phi0 = data[:, 2:]
    phi1 = np.square(data[:, 2:])
    phi2 = np.exp(data[:, 2:])
    phi3 = np.cos(data[:, 2:])
    phi4 = np.ones((len(data), 1))

    phi = np.concatenate((phi0, phi1, phi2, phi3), axis=1)
    y = data[:, 1]

    return phi, y


def main():
    data = ReadFile()
    x, y = x2Phi(data)
    y = y[:, np.newaxis]

    n = x.shape[0]
    m = x.shape[1]
    np.random.seed(101)
    tf.set_random_seed(101)

    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    W = tf.Variable(tf.zeros([1, m]), name='W')
    b = tf.Variable(tf.zeros([1]), name='b')

    learning_rate = 0.1
    training_epochs = 2000

    y_pred = tf.add(tf.multiply(X, W), b)
    cost = tf.reduce_sum(tf.pow(y_pred - Y, 2)) / (2 * n)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            for (_x, _y) in zip(x, y):
                sess.run(optimizer, feed_dict={X: _x, Y: _y})
        training_cost = sess.run(cost, feed_dict={X: x, Y: y})
        weight = sess.run(W)
        bias = sess.run(b)

    print("Training cost =", training_cost, "Weight =", weight, "bias =", bias,
          '\n')

    with open('ParkerTest2.csv', "w") as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow(W)
        writer.writerow(b)


if __name__ == '__main__':
    main()
