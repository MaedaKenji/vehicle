# csv_to_tfrecord_v2.py
import tensorflow as tf
import pandas as pd
import os
import io
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

# Update your label map here
label_map = {
    'Bus': 1,
    'Car': 2,
    'Motorcycle': 3,
    'Pickup': 4,
    'Truck': 5
    # Add more classes if needed
}

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'  # or b'png'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_record(csv_input, image_dir, output_record):
    writer = tf.io.TFRecordWriter(output_record)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created TFRecord: {output_record}')

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    create_record(
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/train/_annotations.csv'),
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/train'),
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/train.record')
    )
    create_record(
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/test/_annotations.csv'),
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/test'),
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/test.record')
    )
    create_record(
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/valid/_annotations.csv'),
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/valid'),
        os.path.join(current_dir, 'Dataset/Grayscale/Vehicles_Detection.v9i.tensorflow/valid.record')
    )