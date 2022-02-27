#coding:utf-8
import os
import random
import sys
import time
import json
import cv2
sys.path.append('/content/MTCNN-Tensorflow/prepare_data')

import tensorflow as tf

from prepare_data.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(imdir,filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    #print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(imdir + '/' + filename, 'ONet')
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #return '%s/%s_%s_%s.tfrecord' % (output_dir, name, net, st)
    #return '%s/train_PNet_landmark.tfrecord' % (output_dir)
    return '%s/plate_landmark.tfrecord' % (output_dir)
    

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
        #andom.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print('lala')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if( i%100 == 0):
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
                sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(dataset_dir,filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')

import copy
def get_dataset(image_dir,net):
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    #item = 'imglists/PNet/train_%s_landmark.txt' % net
    # item = '%s/neg_%s.txt' % (net,net)
    # #print(item)
    # dataset_dir = os.path.join(dir, item)
    # imagelist = open(dataset_dir, 'r')
    if net == "PNet":
        target_size = (12,12)
    elif net == "ONet":
        target_size = (48,48)
    dataset = []
    for filename in os.listdir(image_dir):
      if filename.endswith(".json"):
        filepath = os.path.join(image_dir,filename)
        with open(filepath) as json_file:
          file = json.loads(json_file.read())
          data_example = dict()
          bbox = dict()
          data_example['filename'] = file["imagePath"][:-3]+"jpg"
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
          
          #Get image size
          impath = os.path.join(image_dir,data_example['filename'])
          img = cv2.imread(impath)
          h,w,_ = img.shape
          rh,rw = target_size

          labels = file['shapes']
          for label in labels:
            if label['label'] == 'ltop':
              bbox['xltop'] = label['points'][0][0]*rw/w
              bbox['yltop'] = label['points'][0][1]*rh/h
            if label['label'] == 'lbot':
              bbox['xlbot'] = label['points'][0][0]*rw/w
              bbox['ylbot'] = label['points'][0][1]*rh/h
            if label['label'] == 'rtop':
              bbox['xrtop'] = label['points'][0][0]*rw/w
              bbox['yrtop'] = label['points'][0][1]*rh/h
            if label['label'] == 'rbot':
              bbox['xrbot'] = label['points'][0][0]*rw/w
              bbox['yrbot'] = label['points'][0][1]*rh/h
            if label['label'] == 'plate':
              bbox['xmin'] = label['points'][0][0]*rw/w
              bbox['ymin'] = label['points'][0][1]*rh/h
              bbox['xmax'] = label['points'][1][0]*rw/w
              bbox['ymax'] = label['points'][1][1]*rh/h
          data_example['bbox'] = bbox
          data_example_2 = copy.deepcopy(data_example)
          data_example_2['label'] = -2
          dataset.append(data_example)
          dataset.append(data_example_2)

    return dataset


if __name__ == '__main__':
    dir = '/content/White_Yellow_LPR_cropped'
    net = 'ONet'
    output_directory = '/content/MTCNN-Tensorflow'
    run(dir, net, output_directory, shuffling=True)
