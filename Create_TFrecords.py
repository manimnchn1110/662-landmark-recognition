import tensorflow as tf
import pandas as pd
from PIL import Image


df = pd.read_csv('train.csv')

# First_100_training_set
folder_path = './train' + '/'
writer = tf.python_io.TFRecordWriter("Landmark_train.tfrecords")
for i in range(100):
    img_path = folder_path + df.id[i] + '.jpeg'
    output = int(df.landmark_id[i])
    img = Image.open(img_path)
    img_raw = img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[output])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())

writer.close()
