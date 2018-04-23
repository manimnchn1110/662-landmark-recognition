
import tensorflow as tf


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
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5
    label = tf.cast(features['label'], tf.int64)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                 batch_size=batch_size,
                                                 capacity=whole_sample_size,
                                                 min_after_dequeue=0,
                                                 num_threads=1
                                                 )
    return img_batch,label_batch