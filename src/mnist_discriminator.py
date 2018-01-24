import tensorflow as tf
import numpy as np
import csv
from utils import *

class DiscriminatorModelMixin(object):

    def _accuracy(model, mnist_model, dataset, labels):
        train_pr = mnist_model.proba.eval(feed_dict={
            mnist_model.x: dataset.images,
            mnist_model.y_: dataset.labels,
            mnist_model.keep_prob:1.0
        })
        return model.accuracy.eval(feed_dict={
            model.x: train_pr,
            model.y_: labels,
            model.keep_prob:1.0
        })

    def _batch_accuracy(model, probas, batch_labels):
        return model.accuracy.eval(feed_dict={
            model.x: probas, 
            model.y_: batch_labels,
            model.keep_prob:1.0
        })

    def train(model, mnist_model, mnist_model_checkpoint, mnist_training_dataset, mnist_non_training_dataset, name, n = 100000, restore_point = None):
        test_batch_labels = np.array([(1,0)]*50)
        train_batch_labels = np.array([(0,1)]*50)

        HEADER = ["time (h)", "epoch", "batch acc.", "training acc.", "nontraining acc.", "save path"]
        SAVE_STEP = 501

        TRAIN_RATE_BEGIN = 0.001
        TRAIN_RATE_END = 0.00001
        TRAIN_RATE_EXPONENT = (TRAIN_RATE_BEGIN*1./TRAIN_RATE_END)**(SAVE_STEP*1./n)
        with tf.Session() as sess, open(STATS.format(name), 'a' if restore_point is not None else 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            if restore_point is not None:
                model.saver.restore(sess, restore_point)
            else:
                model._initialize(sess)
                csvwriter.writerow(printtable(TABLEWIDTH, *HEADER))
            timer = Timer().start()
            model.set_train_rate(TRAIN_RATE_BEGIN)
            mnist_model.saver.restore(sess, mnist_model_checkpoint)
            
            for i in range(n):
                # do training or not training data every other batch
                if i%2:
                    # train the model on some training data
                    batch = mnist_training_dataset.next_batch(50)
                    blabels = train_batch_labels
                else:
                    # train the model on some non-training data
                    batch = mnist_non_training_dataset.next_batch(50)
                    blabels = test_batch_labels
                probas = mnist_model.proba.eval(feed_dict={
                    mnist_model.x:batch[0],
                    mnist_model.y_:batch[1],
                    mnist_model.keep_prob:1.0})
                if not i%SAVE_STEP:
                    batch_accuracy = model._batch_accuracy(probas, blabels)
                model.train_step.run(feed_dict={
                    model.x: probas, 
                    model.y_: blabels,
                    model.keep_prob: 0.5
                })
                if not i%SAVE_STEP:
                    model.set_train_rate(model.train_rate*TRAIN_RATE_EXPONENT)
                    if not i%(SAVE_STEP*10):
                        training_accuracy = model._accuracy(mnist_model, mnist_training_dataset, np.array([(0,1)]*len(mnist_training_dataset.images)))
                        nontraining_accuracy = model._accuracy(mnist_model, mnist_non_training_dataset, np.array([(1,0)]*len(mnist_non_training_dataset.images)))
                    else:
                        training_accuracy = nontraining_accuracy = 0
                 
                    save_path = model.saver.save(sess, CHECKPOINT.format(name), global_step=i)
                    info_string = printtable(TABLEWIDTH, "{:0.5f}".format(timer.time()),mnist_training_dataset.epochs_completed, batch_accuracy, training_accuracy, nontraining_accuracy, save_path)
                    csvwriter.writerow(info_string)
                    csvfile.flush()
                  

class Discriminator(InitializableMixin, SavableMixin, DiscriminatorModelMixin):
    def __init__(self):
        self.train_rate = 1
        self._init_graph()
        self._init_saver()
    
    def _init_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 10])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.W1 = tf.Variable(tf.random_normal([10,100],0,0.1))
        self.b1 = tf.Variable(tf.random_normal([100],0,0.1))
        self.logits1 = tf.matmul(self.x,self.W1) + self.b1

        self.bent1 = tf.nn.relu(self.logits1)
        
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop1 = tf.nn.dropout(self.bent1, self.keep_prob)
        
        self.W2 = tf.Variable(tf.random_normal([100,10],0,0.1))
        self.b2 = tf.Variable(tf.random_normal([10],0,0.1))
        self.logits2 = tf.matmul(self.drop1,self.W2) + self.b2

        self.bent2 = tf.nn.relu(self.logits2)
        
        self.drop2 = tf.nn.dropout(self.bent2, self.keep_prob)

        self.W3 = tf.Variable(tf.random_normal([10,2],0,0.1))
        self.b3 = tf.Variable(tf.random_normal([2],0,0.1))
        self.logits = tf.matmul(self.drop2,self.W3) + self.b3

        self.proba = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))

        self.set_train_rate(1)
        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
    def set_train_rate(self, rate):
        self.train_rate = rate
        self.train_step = tf.train.GradientDescentOptimizer(rate).minimize(self.loss)

