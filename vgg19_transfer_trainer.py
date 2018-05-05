import tensorflow as tf
import pandas as pd
import numpy as np
from functools import reduce
from decode import read_and_decode

df = pd.read_csv('./train.csv')

#Global Variable Setup
train_whole_sample_size = 977468
test_whole_sample_size = 244385
gesture_class = len(df.landmark_id.unique())
train_batch_size = 32
test_batch_size = 32
train_path = "./Landmark_train.tfrecords"
graph_path = "./tensorboard"
cnn_model_save_path = "./cnn_model/cnn_model.ckpt"

train_path = "./landmark_train.tfrecords"
test_path = "./landmark_test.tfrecords"
graph_path = "./tensorboard"
cnn_model_save_path = "./cnn_model/cnn_model.ckpt"

#vgg16 pretrained model
data_dict = np.load("./vgg19.npy", encoding='latin1').item()

def convert_rgb_bgr(rgb):
    VGG_MEAN = [103.939, 116.779, 123.68]
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
    return bgr

### Layer Setup Function
def weight_variable(shape, f_name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial, name=f_name)

def bias_variable(shape: object, f_name: object) -> object:
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=f_name)

def Conv2d_Filter(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pooling_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

def get_weight_bias(name):
    global weight, bias
    if data_dict is not None:
        weight_name = "_weight"
        bias_name = "_bias"
        weight = tf.Variable(data_dict[name][0], name = name + weight_name)
        bias =  tf.Variable(data_dict[name][1], name = name + bias_name)
    return weight, bias


sess = tf.InteractiveSession()
### create graph placement
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="images")
x = convert_rgb_bgr(x)
y = tf.placeholder(tf.float32, shape=[None, gesture_class, ], name="labels")

### Layer Setup Function
def weight_variable(shape, f_name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial, name=f_name)


def bias_variable(shape, f_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=f_name)


def Conv2d_Filter(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pooling_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

def get_weight_bias(name):
    if data_dict is not None:
        weight_name = "_weight"
        bias_name = "_bias"
        weight = tf.Variable(data_dict[name][0], name = name + weight_name)
        bias =  tf.Variable(data_dict[name][1], name = name + bias_name)
    return weight, bias