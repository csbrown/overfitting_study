import time
import csv
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from utils import *
import os
import pickle
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys
from cifar100_input import *

class Cifar100ShallowConvolutionalModel(TrainableMixin):
    def __init__(self, x=None, y_=None, summaries=True):
        self._init_graph(x,y_,summaries)
        self._init_saver()

    def _init_graph(self,x=None,y_=None,summaries=True):
        with tf.name_scope("cifar100_scm"):
            with tf.name_scope("meta_variables"):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.learning_rate = tf.Variable(0.001, name="learning_rate", trainable=False)

            with tf.name_scope("layer_0_input"):
                self.x = x if x is not None else tf.placeholder(tf.float32, shape=[None,IMAGE_SIZE*IMAGE_SIZE*3], name="input")
                self.y_ = y_ if y_ is not None else tf.placeholder(tf.float32, shape=[None, 100], name="expected_output")
                self.x_image = tf.reshape(self.x, [-1, IMAGE_SIZE, IMAGE_SIZE, 3], name="x_1d_to_3d")

            with tf.name_scope("layer_1_conv"): 
                layer_1_in_dim = [IMAGE_SIZE, IMAGE_SIZE, 3]
                self.W_conv1 = weight_variable([5, 5, 3, 32], name="weights")
                self.b_conv1 = bias_variable([32], name="bias")
                self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1, name="relu")
                self.h_pool1 = max_pool_2x2(self.h_conv1, name="max_pool")
                layer_1_out_dim = [int(layer_1_in_dim[0]/2), int(layer_1_in_dim[1]/2), 32]

            with tf.name_scope("layer_2_conv"):
                layer_2_in_dim = layer_1_out_dim
                self.W_conv2 = weight_variable([5, 5, 32, 64], name="weights")
                self.b_conv2 = bias_variable([64], name="bias")
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2, name="relu")
                self.h_pool2 = max_pool_2x2(self.h_conv2, name="max_pool")
                layer_2_out_dim = [int(layer_2_in_dim[0]/2), int(layer_2_in_dim[1]/2), 64]

            with tf.name_scope("layer_3_conv"):
                layer_3_in_dim = layer_2_out_dim
                self.W_conv3 = weight_variable([5, 5, 64, 128], name="weights")
                self.b_conv3 = bias_variable([128], name="bias")
                self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3, name="relu")
                self.h_pool3 = max_pool_2x2(self.h_conv3, name="max_pool")
                layer_3_out_dim = [int(layer_3_in_dim[0]/2), int(layer_3_in_dim[1]/2), 128]

            with tf.name_scope("layer_4_fc"):
                layer_4_in_dim = layer_3_out_dim
                layer_4_in_flat_dim = [np.prod(layer_4_in_dim)]
                self.in_flat_fc1 = tf.reshape(self.h_pool3, [-1] + layer_4_in_flat_dim, name="conv_to_fc_flattener")
                self.W_fc1 = weight_variable(layer_4_in_flat_dim+[1024], name="weights")
                self.b_fc1 = bias_variable([1024], name="bias")
                self.h_fc1 = tf.nn.relu(tf.matmul(self.in_flat_fc1, self.W_fc1) + self.b_fc1, name="relu")
                self.drop_fc1 = tf.placeholder_with_default(1.0, [], "dropout_rate")
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.drop_fc1, name="dropout")
                layer_4_out_dim = [1024]

            with tf.name_scope("layer_5_fc"):
                layer_5_in_dim = layer_4_out_dim
                self.W_fc2 = weight_variable(layer_5_in_dim + [100])
                self.b_fc2 = bias_variable([100])
                self.logits = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
                layer_5_out_dim = [100]

            with tf.name_scope("outputs"):
                self.proba = tf.nn.softmax(self.logits, name="proba")
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits), name="loss")
      
            with tf.name_scope("summaries"):
                self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
                self.summaries = {}
                if summaries:
                    self.summaries = {
                        "batch acc.": tf.summary.scalar("batch_acc", self.accuracy)
                    }

            with tf.name_scope("training"): 
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="optimizer")
                self.grads = self.optimizer.compute_gradients(self.loss)
                self.grad_application = self.optimizer.apply_gradients(self.grads, global_step = self.global_step)
                with tf.control_dependencies([self.grad_application] + list(self.summaries.values())):
                    self.train_step = tf.no_op(name="train_step")
