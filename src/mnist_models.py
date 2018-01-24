import time
import csv
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from utils import *
import os

mnist = input_data.read_data_sets(os.path.abspath('../data/MNIST_data'), one_hot=True)

def mnist_batch_train_accuracy(model):
    BATCH_SIZE = 1000
    acc = []
    for i in range(int(len(mnist.train.images)/BATCH_SIZE)):
        acc.append( 
            (
                model.accuracy.eval(feed_dict={
                    model.x: mnist.train.images[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                    model.y_: mnist.train.labels[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                    model.keep_prob: 1
                }),
                len(mnist.train.images[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            )
        )
            
    return sum(map(lambda x: x[0]*x[1], acc))*1./len(mnist.train.images)


class MnistModelMixin(object):
    def train(model, name, n = 100000, restore_point = None):
        with tf.Session() as sess, open(STATS.format(name), 'a' if restore_point is not None else 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            if restore_point is not None:
                model.saver.restore(sess, restore_point)
            else:
                model._initialize(sess)
                csvwriter.writerow(printtable(TABLEWIDTH, "time (h)", "epoch", "step", "batch acc", "train acc", "test acc"))
            timer = Timer().start()
            for i in range(n):
                batch = mnist.train.next_batch(50)
                if i % 500 == 0:
                    batch_accuracy = train_accuracy = test_accuracy = 0
                    if i % 10000 == 0:
                        batch_accuracy = model.accuracy.eval(feed_dict={
                            model.x: batch[0], model.y_: batch[1], model.keep_prob: 1.0})
                        train_accuracy = mnist_batch_train_accuracy(model)
                        test_accuracy = model.accuracy.eval(feed_dict={
                            model.x: mnist.test.images, model.y_: mnist.test.labels, model.keep_prob: 1.0})
                    info_string = printtable(TABLEWIDTH, "{:0.5f}".format(timer.time()),mnist.train.epochs_completed, i, batch_accuracy, train_accuracy, test_accuracy)
                    csvwriter.writerow(info_string)
                    csvfile.flush()
                    save_path = model.saver.save(sess, CHECKPOINT.format(name), global_step=i)

                model.train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.5})

            batch_accuracy = model.accuracy.eval(feed_dict={
                model.x: batch[0], model.y_: batch[1], model.keep_prob: 1.0})
            train_accuracy = mnist_batch_train_accuracy(model)
            test_accuracy = model.accuracy.eval(feed_dict={
              model.x: mnist.test.images, model.y_: mnist.test.labels, model.keep_prob: 1.0})
            csvwriter.writerow(printtable(TABLEWIDTH, "{:0.5f}".format(timer.time()),mnist.train.epochs_completed, i, batch_accuracy, train_accuracy, test_accuracy))





class MnistConvolutionalModel(MnistModelMixin, InitializableMixin, SavableMixin):
    def __init__(self):
        self._init_graph()
        self._init_saver()

    def _init_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None,784])
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_fc1 = weight_variable([7*7*64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.logits = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        self.proba = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
   
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

 
class MnistLogisticModel(MnistModelMixin, InitializableMixin, SavableMixin):
    def __init__(self):
        self._init_graph()
        self._init_saver()

    def _init_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.W = weight_variable([784,10])
        self.b = bias_variable([10])
        self.logits = tf.matmul(self.x,self.W) + self.b
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
        self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

