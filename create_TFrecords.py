import tensorflow as tf
import pandas as pd
from PIL import Image

def create_TFrecord(df, file_name):
    folder_path = './train' + '/'
    writer = tf.python_io.TFRecordWriter(file_name)
    error = 0
    sample_count = 0
    error_li = []

    for i in range(len(df)):
        try:
            img_path = folder_path + df.id.iloc[i] + '.jpeg'
            img = None
            img = Image.open(img_path)
            if img is not None:
                label = int(df.landmark_id.iloc[i])
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
                sample_count += 1
                print('writed')
        except:
            print('img not exists!')
            error_li.append(str(df.id.iloc[i]))
            error += 1


    writer.close()
    return sample_count,error, error_li

training_set = pd.read_csv('train.csv')
train_sample_count,train_error, train_error_li = create_TFrecord(training_set, "landmark_training_set.tfrecords")

test_set = training_set.sample(2000, random_state=222, replace=False)
test_sample_count,test_error, test_error_li = create_TFrecord(test_set, "landmark_test_set.tfrecords")