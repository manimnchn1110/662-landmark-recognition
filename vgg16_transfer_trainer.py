import tensorflow as tf
import pandas as pd
import numpy as np
from decode import read_and_decode

#Global Variable Setup
df = pd.read_csv('./train.csv')
train_whole_sample_size = 977468
test_whole_sample_size = 244385
gesture_class = len(df.landmark_id.unique())
train_batch_size = 22
test_batch_size = 22
train_path = "./landmark_train.tfrecords"
test_path = "./landmark_test.tfrecords"
graph_path = "./tensorboard"
cnn_model_save_path = "./cnn_model/cnn_model.ckpt"

#vgg16 pretrained model
data_dict = np.load("./vgg16.npy", encoding='latin1').item()

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


##conv1_1
##224 224 3
##224 224 64
with tf.name_scope('conv1_1'):
    #W_conv1 = weight_variable([3,3,3,64], 'W_conv1')
    #b_conv1 = bias_variable([64], 'b_conv1')
    W_conv1_1, b_conv1_1 = get_weight_bias('conv1_1')
    with tf.name_scope('h_conv1'):
        h_conv1_1 = tf.nn.relu(Conv2d_Filter(x, W_conv1_1) + b_conv1_1)

##224 224 64
##conv1_2
with tf.name_scope('conv1_2'):
    W_conv1_2, b_conv1_2 = get_weight_bias('conv1_2')
    with tf.name_scope('h_conv2'):
        h_conv1_2 = tf.nn.relu(Conv2d_Filter(h_conv1_1, W_conv1_2) + b_conv1_2)

##112 112 64
##pool1
with tf.name_scope('Pool1'):
    h_pool1 = max_pooling_2x2(h_conv1_2, 'pool1')

##112 112 64
##112 112 128
##conv2_1
with tf.name_scope('conv2_1'):
    W_conv2_1, b_conv2_1 = get_weight_bias('conv2_1')
    with tf.name_scope('h_conv2'):
        h_conv2_1 = tf.nn.relu(Conv2d_Filter(h_pool1, W_conv2_1) + b_conv2_1)

##112 112 128
##conv2_2
with tf.name_scope('conv2_2'):
    W_conv2_2, b_conv2_2 = get_weight_bias('conv2_2')
    with tf.name_scope('h_conv2'):
        h_conv2_2 = tf.nn.relu(Conv2d_Filter(h_conv2_1, W_conv2_2) + b_conv2_2)

##56 56 128
##pool2
with tf.name_scope('Pool2'):
    h_pool2 = max_pooling_2x2(h_conv2_2, 'pool2')

##56 56 128
##56 56 256
##conv3_1
with tf.name_scope('conv3_1'):
    W_conv3_1, b_conv3_1 = get_weight_bias('conv3_1')
    with tf.name_scope('h_conv3_1'):
        h_conv3_1 = tf.nn.relu(Conv2d_Filter(h_pool2, W_conv3_1) + b_conv3_1)

##56 56 256
##conv3_2
with tf.name_scope('conv3_2'):
    W_conv3_2, b_conv3_2 = get_weight_bias('conv3_2')
    with tf.name_scope('h_conv3_2'):
        h_conv3_2 = tf.nn.relu(Conv2d_Filter(h_conv3_1, W_conv3_2) + b_conv3_2)

##56 56 256
##conv3_3
with tf.name_scope('conv3_3'):
    W_conv3_3, b_conv3_3 = get_weight_bias('conv3_3')
    with tf.name_scope('h_conv3_3'):
        h_conv3_3 = tf.nn.relu(Conv2d_Filter(h_conv3_2, W_conv3_3) + b_conv3_3)

##28 28 256
##pool3
with tf.name_scope('Pool3'):
    h_pool3 = max_pooling_2x2(h_conv3_3, 'pool3')

##28 28 256
## 28 28 512
##conv4_1
with tf.name_scope('conv4_1'):
    W_conv4_1, b_conv4_1 = get_weight_bias('conv4_1')
    with tf.name_scope('h_conv4_1'):
        h_conv4_1 = tf.nn.relu(Conv2d_Filter(h_pool3, W_conv4_1) + b_conv4_1)

##28 28 512
##conv4_2
#with tf.name_scope('conv4_2'):
#    W_conv4_2, b_conv4_2 = get_weight_bias('conv4_2')
#    with tf.name_scope('h_conv4_2'):
#        h_conv4_2 = tf.nn.relu(Conv2d_Filter(h_conv4_1, W_conv4_2) + b_conv4_2)

##28 28 512
##conv4_3
#with tf.name_scope('conv4_3'):
#    W_conv4_3, b_conv4_3 = get_weight_bias('conv4_3')
#    with tf.name_scope('h_conv4_3'):
#        h_conv4_3 = tf.nn.relu(Conv2d_Filter(h_conv4_2, W_conv4_3) + b_conv4_3)

## 14 14 512
##pool4
with tf.name_scope('Pool4'):
    h_pool4 = max_pooling_2x2(h_conv4_1, 'pool4')

## 14 14 512
##conv5_1
with tf.name_scope('conv5_1'):
    W_conv5_1, b_conv5_1 = get_weight_bias('conv5_1')
    with tf.name_scope('h_conv5_1'):
        h_conv5_1 = tf.nn.relu(Conv2d_Filter(h_pool4, W_conv5_1) + b_conv5_1)

##  14 14 512
##conv5_2
#with tf.name_scope('conv5_2'):
#    W_conv5_2, b_conv5_2 = get_weight_bias('conv5_2')
#    with tf.name_scope('h_conv5_2'):
#        h_conv5_2 = tf.nn.relu(Conv2d_Filter(h_conv5_1, W_conv5_2) + b_conv5_2)

## 14 14 512
##conv5_3
#with tf.name_scope('conv5_3'):
#    W_conv5_3, b_conv5_3 = get_weight_bias('conv5_3')
#    with tf.name_scope('h_conv5_3'):
#        h_conv5_3 = tf.nn.relu(Conv2d_Filter(h_conv5_2, W_conv5_3) + b_conv5_3)

##7 7 512
##pool5
with tf.name_scope('Pool5'):
    h_pool5 = max_pooling_2x2(h_conv5_1, 'pool5')

## 1 25088
##fc6
with tf.name_scope("fc6"):
    W_fc6, b_fc6 = get_weight_bias('fc6')

    with tf.name_scope('Pool6_flat'):
        h_pool6_flat = tf.reshape(h_pool5, [-1, 25088])

    with tf.name_scope('h_fc6'):
        h_fc6 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc6) + b_fc6)

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob, name="h_fc6_drop")

## 1 4096
##fc7
with tf.name_scope("fc7"):
    W_fc7, b_fc7 = get_weight_bias('fc7')

    with tf.name_scope('Pool5_flat'):
        h_pool7_flat = tf.reshape(h_fc6_drop, [-1, 4096])

    with tf.name_scope('h_fc1'):
        h_fc7 = tf.nn.relu(tf.matmul(h_pool7_flat, W_fc7) + b_fc7)

## 1 classes
##fc8
with tf.name_scope('my_fc8'):
    W_fc8 = weight_variable([4096, gesture_class], 'my_fc8')
    b_fc8 = bias_variable([gesture_class], 'my_fc8')
    with tf.name_scope('fc8_softmax'):
        my_prediction = tf.nn.softmax(tf.matmul(h_fc7, W_fc8) + b_fc8, name="my_prediction")

# Define training step
with tf.name_scope('Corss_Entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(my_prediction), reduction_indices=[1]))
    tf.summary.scalar('corss_entropy', cross_entropy)

with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name="train_step")

# Define testing step
correct_prediction = tf.equal(tf.argmax(my_prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# import training set
img_train_batch, labels_train_batch = read_and_decode(train_path, train_batch_size, train_whole_sample_size)
train_label = tf.one_hot(labels_train_batch, gesture_class, 1, 0)

img_test_batch, labels_test_batch = read_and_decode(test_path, test_batch_size, test_whole_sample_size)
test_label = tf.one_hot(labels_test_batch, gesture_class, 1, 0)

###Start to Train
with tf.Session(config=tf.ConfigProto(
    device_count={'CPU':8},
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1
)) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, cnn_model_save_path)

    train_writer = tf.summary.FileWriter(graph_path, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    img_train, label_train = sess.run([img_train_batch, train_label])
    img_test, label_test = sess.run([img_test_batch, test_label])

    acc = sess.run(accuracy, feed_dict={x: img_test, y: label_test, keep_prob: 1.0})
    print("Accuracy before train: " + str(acc))
    max_acc = acc
    for i in range(101):

        print("Trained:", i)
        sess.run(train_step, feed_dict={x: img_train, y: label_train, keep_prob: 1.0})

        if (i % 5) == 0:
            print("Trained:", i)

            acc = sess.run(accuracy, feed_dict={x: img_test, y: label_test, keep_prob: 1.0})
            print("Itsers = " + str(i) + "  Accuracy: " + str(acc))

            if max_acc < acc:
                max_acc = acc
                saver.save(sess, save_path=cnn_model_save_path)
                print('Model updated! Accuracy:', max_acc)
                summay = sess.run(merged, feed_dict={x: img_test, y: label_test, keep_prob: 1.0})
                train_writer.add_summary(summay, i)
                print('Sumarry updated!')

            if max_acc > 0.90:
                break

    train_writer.close()
    coord.request_stop()
    coord.join(threads)
    sess.close()


