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
from cifar10_input import *


class Cifar10ShallowConvolutionalModel(TrainableMixin):
    def __init__(self, x=None, y_=None, summaries=True, trainable=True):
        self._init_graph(x,y_,summaries,trainable)
        self._init_saver()

    def _init_graph(self,x=None,y_=None,summaries=True,trainable=True):
        with tf.name_scope("cifar10_scm"):
            with tf.name_scope("meta_variables"):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.learning_rate = tf.Variable(0.001, name="learning_rate", trainable=False)

            with tf.name_scope("layer_0_input"):
                layer_0_output_dim = [IMAGE_SIZE, IMAGE_SIZE, 3]
                self.input_dim = [np.product(layer_0_output_dim)]
                self.output_dim = [10]
                self.x = x if x is not None else tf.placeholder(tf.float32, shape=[None] + self.input_dim, name="input")
                self.y_ = y_ if y_ is not None else tf.placeholder(tf.float32, shape=[None] + self.output_dim, name="expected_output")
                self.x_image = tf.reshape(self.x, [-1] + layer_0_output_dim, name="x_1d_to_3d")


            with tf.name_scope("layer_1_conv"): 
                layer_1_in_dim = layer_0_output_dim
                self.W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.1), name="weights", trainable=trainable)
                self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name="bias", trainable=trainable)
                self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1, name="relu")
                self.h_pool1 = max_pool_2x2(self.h_conv1, name="max_pool")

                self.weight_decay_layer_1 = tf.Variable(0,trainable=False,name="weight_decay_rate")
                self.layer_1_loss = tf.multiply(tf.nn.l2_loss(self.W_conv1), self.weight_decay_layer_1, name="weight_loss")

                layer_1_out_dim = [int(layer_1_in_dim[0]/2), int(layer_1_in_dim[1]/2), 32]

            with tf.name_scope("layer_2_conv"):
                layer_2_in_dim = layer_1_out_dim
                self.W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="weights", trainable=trainable)
                self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]), name="bias", trainable=trainable)
                self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2, name="relu")
                self.h_pool2 = max_pool_2x2(self.h_conv2, name="max_pool")

                self.weight_decay_layer_2 = tf.Variable(0,trainable=False,name="weight_decay_rate")
                self.layer_2_loss = tf.multiply(tf.nn.l2_loss(self.W_conv2), self.weight_decay_layer_2, name="weight_loss")

                layer_2_out_dim = [int(layer_2_in_dim[0]/2), int(layer_2_in_dim[1]/2), 64]

            with tf.name_scope("layer_3_conv"):
                layer_3_in_dim = layer_2_out_dim
                self.W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1), name="weights", trainable=trainable)
                self.b_conv3 = tf.Variable(tf.constant(0.1,shape=[128]), name="bias", trainable=trainable)
                self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3, name="relu")
                self.h_pool3 = max_pool_2x2(self.h_conv3, name="max_pool")

                self.weight_decay_layer_3 = tf.Variable(0,trainable=False,name="weight_decay_rate")
                self.layer_3_loss = tf.multiply(tf.nn.l2_loss(self.W_conv2), self.weight_decay_layer_3, name="weight_loss")

                layer_3_out_dim = [int(layer_3_in_dim[0]/2), int(layer_3_in_dim[1]/2), 128]

            with tf.name_scope("layer_4_fc"):
                layer_4_in_dim = layer_3_out_dim
                layer_4_in_flat_dim = [np.prod(layer_4_in_dim)]
                self.in_flat_fc1 = tf.reshape(self.h_pool3, [-1] + layer_4_in_flat_dim, name="conv_to_fc_flattener")
                self.W_fc1 = tf.Variable(tf.truncated_normal(layer_4_in_flat_dim+[1024], stddev=0.1), name="weights", trainable=trainable)
                self.b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]), name="bias", trainable=trainable)
                self.h_fc1 = tf.nn.relu(tf.matmul(self.in_flat_fc1, self.W_fc1) + self.b_fc1, name="relu")
                self.drop_fc1 = tf.Variable(1.0, name="dropout_rate",trainable=False)
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.drop_fc1, name="dropout")
                layer_4_out_dim = [1024]

            with tf.name_scope("layer_5_fc"):
                layer_5_in_dim = layer_4_out_dim
                self.W_fc2 = tf.Variable(tf.truncated_normal(layer_5_in_dim + [10], stddev=0.1), name="weights", trainable=trainable)
                self.b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]), name="bias", trainable=trainable)
                self.logits = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
                layer_5_out_dim = [10]

            with tf.name_scope("outputs"):
                self.proba = tf.nn.softmax(self.logits, name="proba")
                self.weight_decay_layer_2 = tf.Variable(0,trainable=False)
                self.weight_decay_layer_3 = tf.Variable(0,trainable=False)
                self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits), name="cross_entropy_loss")
                
      
            with tf.name_scope("summaries"):
                self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
                self.summaries = {}
                if summaries:
                    self.summaries = {
                        "batch acc.": tf.summary.scalar("batch_acc", self.accuracy)
                    }

            if trainable:
                with tf.name_scope("training"): 
                    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="optimizer")
                    self.grads = self.optimizer.compute_gradients(self.loss)
                    self.grad_application = self.optimizer.apply_gradients(self.grads, global_step = self.global_step)
                    with tf.control_dependencies([self.grad_application] + list(self.summaries.values())):
                        self.train_step = tf.no_op(name="train_step")
