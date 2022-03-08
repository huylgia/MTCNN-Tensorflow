#thay vào train_onet.py
#coding:utf-8
from train_models.mtcnn_model import O_Net1, O_Net2
from train_models.train import train
import argparse
import os


def train_ONet(tfrecord_path, data_dir, prefix, end_epoch,lr, pretrained_model):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    if "CustomLayer" in prefix:
        net_factory = O_Net1
    else:
        net_factory = O_Net2
    train(net_factory, prefix, end_epoch, tfrecord_path, data_dir, base_lr=lr, pretrained_model=pretrained_model)
def get_parser():
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument(
        "--model_name",
        default='ONet_landmark',
        help="name of model",
    )
    parser.add_argument(
        "--model_dir",
        default='/content/MTCNN-Tensorflow/data/MTCNN_model',
        help="model directory",
    )
    parser.add_argument(
        "--data_dir",
        default='/content/LPR_MTCNN_GO1_0_1_Part0',
        help="dataset directory",
    )
    parser.add_argument(
        "--tfrecord_path",
        default='/content/MTCNN-Tensorflow/plate_landmark.tfrecord_shuffle',
        help="path to tfrecord dataset",
    )
    parser.add_argument(
      '--pretrained_model', type=str, 
      default='',
      help='Load a pretrained model before training starts.'
    )
    parser.add_argument(
      '--end_epoch', type=int, 
      default=8000,
      help='number of epoch for training'
    )
    parser.add_argument(
      '--lr', type=float, 
      default=0.00001,
      help='number of epoch for training'
    )
    return parser
if __name__ == '__main__':
    args = get_parser().parse_args()
    #model_name = 'MTCNN'
    model_dir = args.model_dir
    model_path = os.path.join(model_dir,args.model_name,"ONet")
    tfrecord_path = args.tfrecord_path
    data_dir = args.data_dir
    end_epoch = args.end_epoch
    lr = args.lr
    pretrained_model = args.pretrained_model
    if pretrained_model:
      pretrained_model = os.path.expanduser(args.pretrained_model)
    train_ONet(tfrecord_path, data_dir, model_path, end_epoch, lr, pretrained_model)
