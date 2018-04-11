##CNN - Landmark Recognition Trainer and Saver
import tensorflow as tf
import pandas as pd

df = pd.read_csv('train.csv')
#Global Variable Setup
train_whole_sample_size = 100
test_whole_sample_size = 100
gesture_class = len(df.landmark_id[0:100].unique())
train_batch_size = 10
test_batch_size = 100
image_size = 224
train_path = "./Landmark_train.tfrecords"
graph_path = "./tensorboard"
cnn_model_save_path = "./cnn_model/cnn_model.ckpt"


def read_and_decode(filename, batch_size, whole_sample_size):
    global gesture_class, image_size
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [image_size, image_size, 3])
    img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5
    label = tf.cast(features['label'], tf.int64)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                 batch_size=batch_size,
                                                 capacity=whole_sample_size,
                                                 min_after_dequeue=50,
                                                 num_threads=2
                                                 )
    return img_batch,label_batch

#training set
img_train_batch, labels_train_batch = read_and_decode(train_path, train_batch_size, train_whole_sample_size)
train_label = tf.one_hot(labels_train_batch, gesture_class, 1, 0)


#test set
img_test_batch, labels_test_batch = read_and_decode(train_path, test_batch_size, test_whole_sample_size)
test_label = tf.one_hot(labels_test_batch, gesture_class, 1, 0)


### create session
sess = tf.InteractiveSession()

### create graph placement
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="images")
y = tf.placeholder(tf.float32, shape=[None, gesture_class,], name="labels")

### Layer Setup Function
def weight_variable(shape, f_name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial, name=f_name)


def bias_variable(shape, f_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=f_name)


def Conv2d_Filter(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


### 1)Layer 1: conv 3x3 64 (RELU)
'''- Input: 224x224x3
- Type: Conv
- size: 3x3
- channel: 64
- strides: 1
- Output: 224x224x64
'''
with tf.name_scope('Conv1'):
    W_conv1 = weight_variable([3,3,3,64], 'W_conv1')
    b_conv1 = bias_variable([64], 'b_conv1')
    with tf.name_scope('h_conv1'):
        h_conv1 = tf.nn.relu(Conv2d_Filter(x, W_conv1) + b_conv1)

### 2)Layer 1: max pooling 2x2 2
'''
- Input: 224x224x64
- pool: 2x2
- output: 112x112x64
'''
with tf.name_scope('Pool1'):
    h_pool1 = max_pooling_2x2(h_conv1)

### 3)Layer 2: conv 3x3 128 (RELU)

'''
- Input: 112x112x64
- Type: Conv
- size: 3x3
- channel: 128
- strides: 1
- Output: 112x112x128
'''

with tf.name_scope('Conv2'):
    W_conv2 = weight_variable([3,3,64,128], 'W_conv2')
    b_conv2 = bias_variable([128], 'b_conv2')
    with tf.name_scope('h_conv2'):
        h_conv2 = tf.nn.relu(Conv2d_Filter(h_pool1, W_conv2) + b_conv2)

### 4)Layer 2: max pooling 2x2 2
'''
- Input: 112x112x128
- pool: 2x2
- output: 56x56x128
'''
with tf.name_scope('Pool2'):
    h_pool2 = max_pooling_2x2(h_conv2)

### 5)Layer 3: conv 3x3 256 (RELU)
'''
- Input: 56x56x128
- Type: Conv
- size: 3x3
- channel: 256
- strides: 1
- Output: 56x56x256
'''
with tf.name_scope('Conv3'):
    W_conv3 = weight_variable([3,3,128,256], 'W_conv3')
    b_conv3 = bias_variable([256], 'b_conv3')
    with tf.name_scope('h_conv3'):
        h_conv3 = tf.nn.relu(Conv2d_Filter(h_pool2, W_conv3) + b_conv3)
### 6)Layer 3: max pooling 2x2 2
"""
- Input: 56x56x256
- pool: 2x2
- output: 28x28x256
"""
with tf.name_scope('Pool3'):
    h_pool3 = max_pooling_2x2(h_conv3)

### 7)Layer 4: conv 3x3 512 (RELU)
'''
- Input: 28x28x256
- Type: Conv
- size: 3x3
- channel: 512
- strides: 1
- Output: 28x28x512
'''
with tf.name_scope('Conv4'):
    W_conv4 = weight_variable([3,3,256,512], 'W_conv4')
    b_conv4 = bias_variable([512], 'b_conv4')
    with tf.name_scope('h_conv4'):
        h_conv4 = tf.nn.relu(Conv2d_Filter(h_pool3, W_conv4) + b_conv4)
### 8)Layer 4: max pooling 2x2 2
'''
- Input: 28x28x512
- pool: 2x2
- output: 14x14x512
'''

with tf.name_scope('Pool4'):
    h_pool4 = max_pooling_2x2(h_conv4)

### 9)Layer 5: conv 3x3 512 (RELU)
"""
- Input: 14x14x512
- Type: Conv
- size: 3x3
- channel: 512
- strides: 1
- Output: 14x14x512
"""

with tf.name_scope('Conv5'):
    W_conv5 = weight_variable([3,3,512,512], 'W_conv5')
    b_conv5 = bias_variable([512], 'b_conv5')
    with tf.name_scope('h_conv5'):
        h_conv5 = tf.nn.relu(Conv2d_Filter(h_pool4, W_conv5) + b_conv5)
### 10)Layer 5: max pooling 2x2 2
'''
- Input: 14x14x512
- pool: 2x2
- output: 7x7x512
'''
with tf.name_scope('Pool5'):
    h_pool5 = max_pooling_2x2(h_conv5)

### 11)Layer 6: Fully Connected Layer 4096 (ReLU)
### 12)Layer 7: Fully Connected Layer - Dropout Layer (reducing overfitting)
'''
reshape:[1, 7x7x512 = 25088]
output: [1, 4096]
'''
with tf.name_scope('Fc1'):
    W_fc1 = weight_variable([25088, 4096], 'W_fc1')
    b_fc1 = bias_variable([4096], 'b_fc1')

    with tf.name_scope('Pool5_flat'):
        h_pool5_flat = tf.reshape(h_pool5, [-1, 25088])

    with tf.name_scope('h_fc1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32, name="my_keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="my_h_fc1_drop")

###  13)Layer 8: Readout Layer (Softmax Layer)
with tf.name_scope('Fc2'):
    # 第二个全连接层
    W_fc2 = weight_variable([4096, gesture_class], 'W_fc2')
    b_fc2 = bias_variable([gesture_class], 'b_fc2')

    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="my_prediction")

# Define functions and train the model
'''
### Type of optimization: Gradient Descent
- Training process：
1. Calculate Error Rate（corss_entropy）
2. Reduce Error Rate
3. Revised Kernels's weight to reduce Error Rate (corss_entropy)
4. Calculate correct prediction number
5. Calculate accuracy
6. Summary the prediction process
'''
with tf.name_scope('Corss_Entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name="loss")
    tf.summary.scalar('corss_entropy', cross_entropy)

with tf.name_scope('Train_step'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name="train_step")

correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

###Start to Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(graph_path, sess.graph);

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    saver = tf.train.Saver()
    max_acc = 0

    for i in range(1001):

        img_xs, label_xs = sess.run([img_train_batch, train_label])
        sess.run(train_step, feed_dict={x: img_xs, y: label_xs, keep_prob: 0.75})

        if (i % 1) == 0:
            print("The", i, "Train")
            img_test_xs, label_test_xs = sess.run([img_test_batch, test_label])
            acc = sess.run(accuracy, feed_dict={x: img_test_xs, y: label_test_xs, keep_prob: 1.0})
            print("Itsers = " + str(i) + "  Accuracy: " + str(acc))

            summay = sess.run(merged, feed_dict={x: img_test_xs, y: label_test_xs, keep_prob: 1.00})

            train_writer.add_summary(summay, i)

            if max_acc < acc:
                max_acc = acc
                saver.save(sess, save_path=cnn_model_save_path)

            if acc > 0.50:
                break

    train_writer.close()

    coord.request_stop()
    coord.join(threads)
    sess.close()