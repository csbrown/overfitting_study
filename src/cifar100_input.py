import os
import tensorflow as tf
import functools
from urllib.request import urlretrieve
import zipfile
import tarfile
import sys
from utils import *

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32
BYTE = 1
KB = 1024*BYTE
MB = 1024*KB

# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def maybe_download_and_extract():
    main_directory = "../data/"
    data_directory = main_directory+"cifar_100/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    if not os.path.exists(data_directory):
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_data = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")
        os.rename(main_directory+"cifar-100-binary", data_directory)
        os.remove(zip_data)
    train_data = split_data(os.path.join(data_directory, 'train.bin'),10000)
    return (
        train_data[:-1],
        train_data[-1:],
        [os.path.join(data_directory, 'test.bin')]
    )

def split_data(filename, size, suffix=""):
    files = []
    with open(filename,'rb') as orig_data_file:
        data = orig_data_file.read(Cifar100Record.record_bytes)
        j = 0
        while data:
            files.append(filename+suffix+str(j).rjust(3,"0"))
            with open(files[-1],'wb') as new_data_file:
                for i in range(size):
                    new_data_file.write(data)
                    data = orig_data_file.read(Cifar100Record.record_bytes)
            j+=1
    return files
    

class Cifar100Record(object):
    # This class represents a Cifar100 thing that we read from the Cifar100 files.
    label_bytes = 2    # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    image_bytes = height * width * depth
    record_bytes = label_bytes + image_bytes
    def __init__(self, filenames, batch_size=100, epochs=None, distort=False, crop=None, shuffle=True):
        ''' This is going to create the tf graph that reads a record. '''
        
        self.filename_queue = tf.train.string_input_producer(filenames)
        self.dataset = tf.data.FixedLengthRecordDataset(filenames, Cifar100Record.record_bytes, buffer_size=1*MB)
        self.dataset = self.dataset.map(Cifar100Record._parse)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=1*MB)
        cropper = tf.image.resize_image_with_crop_or_pad
        if distort:
            cropper = tf.image.random_crop
            self.dataset = self.dataset.map(Cifar100Record._distort)
        if crop:
            self.dataset = self.dataset.map(functools.partial(Cifar100Record._crop, cropper, *crop))
        self.dataset = self.dataset.map(Cifar100Record._standardize)
        self.dataset = self.dataset.batch(batch_size)
        # If epochs is Falsey, go 'forever'
        FOREVER = tf.int64.max
        epochs = epochs or FOREVER
        self.dataset = self.dataset.repeat(epochs)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.image, self.label = self.iterator.get_next()

    @staticmethod
    def _parse(value):
        record = tf.decode_raw(value, tf.uint8)
        coarse_label = tf.cast(
            tf.strided_slice(record, [0], [1]), tf.int32)
        fine_label = tf.cast(
            tf.strided_slice(record, [1], [2]), tf.int32)
        label = coarse_label + (tf.uint8.max + 1)*fine_label
        label = tf.one_hot(label, 100, on_value=1.0, off_value=0.0)
        label = tf.reshape(label,[100])

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
                tf.strided_slice(record, [Cifar100Record.label_bytes],
                    [Cifar100Record.record_bytes]),
                [Cifar100Record.depth, Cifar100Record.height, Cifar100Record.width])
        # Convert from [depth, height, width] to [height, width, depth].
        uint8image = tf.transpose(depth_major, [1, 2, 0])
        float32image = tf.cast(uint8image, tf.float32)
        return float32image, label

    @staticmethod
    def _crop(cropper, height, width, image, label):
        return cropper(image, height, width), label

    @staticmethod
    def _distort(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        return image, label
       
    @staticmethod
    def _standardize(image, label):
        image = tf.image.per_image_standardization(image)
        return image, label 
