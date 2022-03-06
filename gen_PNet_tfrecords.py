#coding:utf-8
import os
import random
import sys
import time
import json

import tensorflow as tf

from prepare_data.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      filename: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    #print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder('/content/ONet/' + filename, 'PNet')
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    return '%s/train_PNet_landmark.tfrecord' % (output_dir)
    

def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        #random.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print('lala')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 100 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (i+1, len(dataset)))
                #sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, net='PNet'):
    #get file name , label and anotation
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    # item = 'imglists/PNet/train_%s_landmark.txt' % net
    
    # dataset_dir = os.path.join(dir, item)
    # #print(dataset_dir)
    # imagelist = open(dataset_dir, 'r')

    dataset = []
    for filename in os.listdir('/content/ONet'):
      if filename.endswith(".json"):
        with open('/content/ONet/' + filename) as json_file:
          file = json.loads(json_file.read())
          data_example = dict()
          bbox = dict()
          data_example['filename'] = file["imagePath"]
          data_example['label'] = 1
          bbox['xmin'] = 0
          bbox['ymin'] = 0
          bbox['xmax'] = 0
          bbox['ymax'] = 0
          bbox['xltop'] = 0
          bbox['yltop'] = 0
          bbox['xrtop'] = 0
          bbox['yrtop'] = 0
          bbox['xlbot'] = 0
          bbox['ylbot'] = 0
          bbox['xrbot'] = 0
          bbox['yrbot'] = 0
          
          labels = file['shapes']
          for label in labels:
            if label['label'] == 'ltop':
              bbox['xltop'] = label['points'][0][0]
              bbox['yltop'] = label['points'][0][1]
            if label['label'] == 'lbot':
              bbox['xlbot'] = label['points'][0][0]
              bbox['ylbot'] = label['points'][0][1]
            if label['label'] == 'rtop':
              bbox['xrtop'] = label['points'][0][0]
              bbox['yrtop'] = label['points'][0][1]
            if label['label'] == 'rbot':
              bbox['xrbot'] = label['points'][0][0]
              bbox['yrbot'] = label['points'][0][1]
            if label['label'] == 'plate':
              bbox['xmin'] = label['points'][0][0]
              bbox['ymin'] = label['points'][0][1]
              bbox['xmax'] = label['points'][1][0]
              bbox['ymax'] = label['points'][1][1]
          data_example['bbox'] = bbox
          dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = 'DATA/'
    net = 'PNet'
    output_directory = '/content/MTCNN-Tensorflow'
    run(dir, net, output_directory, shuffling=True)
