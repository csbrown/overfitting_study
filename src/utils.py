import time
import numpy as np
import tensorflow as tf
import sys
import os

ROOT = "../models/{}"
CHECKPOINT = "{}/model.ckpt".format(ROOT)
STATS = "{}_stats".format(ROOT)
TABLEWIDTH = 16

def make_dirs(name): 
    try:
        os.makedirs(name)
    except FileExistsError:
        pass

def load_checkpoints(name):
    try:
        checkpoints = sorted(["{}/{}".format(ROOT.format(name),x[:-5]) for x in os.listdir(ROOT.format(name)) if x.endswith(".meta")], 
                     key=lambda x: int(x.split("-")[-1]))
    except FileNotFoundError:
        checkpoints = []
    return checkpoints

class Timer(object):
    def start(self):
        self.start = time.time()
        return self
    def time(self):
        return (time.time() - self.start)/60/60

def printtable(n, *args):
    return list(map(lambda x: str(x).ljust(n," "), args))


def entropy(vec):
    # takes in a vector that sums to 1
    total = 0
    for item in vec:
        total += np.log2(item)*item
    return total

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

class VariablesMixin(object):
    def _gather_variables(model):
        if not hasattr(model, "_variables"):
            model._variables = [model.__dict__[x] for x in model.__dict__ if isinstance(model.__dict__[x], tf.Variable)]

class InitializableMixin(VariablesMixin):
    def _initialize(model, sess):
        model._gather_variables()
        for variable in model._variables:
            sess.run(variable.initializer)

class SavableMixin(VariablesMixin):
    def _init_saver(model):
        model._gather_variables()
        model.saver = tf.train.Saver(model._variables, max_to_keep=None)

class RestorableMixin(InitializableMixin, SavableMixin):
    def _restore_or_init(model, sess, restore_point=None):
        if restore_point is not None:
            model.saver.restore(sess, restore_point)
        else:
            model._initialize(sess)

class TrainableMixin(RestorableMixin):
    def train(model, name, restore_point = None, save_rate = 500):
        with tf.Session() as sess:
            model._restore_or_init(sess, restore_point)
            try:
                while True:
                    sess.run(model.train_step)
                    global_step = model.global_step.eval()
                    if not global_step%save_rate:
                        print("Saving to: ", CHECKPOINT.format(name) + "_" + str(global_step))
                        model.saver.save(sess, CHECKPOINT.format(name), global_step=global_step)
            except KeyboardInterrupt:
                pass

def print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def BatchableData(object):
    def __init__(self, data_x, data_y, shuffle=True):
        self.data_x = data_x
        self.data_y = data_y
        self._data_x = data_x
        self._data_y = data_y
        self.shuffle = shuffle
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(data_x)

    def _shuffle(self):
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._data_x = self.data_x[perm]
        self._data_y = self.data_y[perm]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for each epoch
        if self._index_in_epoch == 0 and self.shuffle:
            self._shuffle()
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if self.shuffle:
                self._shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_x_new_part = self._data_x[start:end]
            data_y_new_part = self._data_y[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
    return self._data_x[start:end], self._data_y[start:end]

