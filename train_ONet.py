#coding:utf-8
from train_models.mtcnn_model import O_Net
from train_models.train import train
import argparse


def train_ONet(base_dir, prefix, end_epoch, display, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    net_factory = O_Net
    train(net_factory, prefix, end_epoch, base_dir, display=display, base_lr=lr)
def get_parser():
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument(
        "--model_name",
        default='ONet_landmark',
        help="name of model ",
    )
    parser.add_argument(
        "--tfrecord_path",
        default='/content/MTCNN-Tensorflow/plate_landmark.tfrecord_shuffle',
        help="path to tfrecord dataset",
    )
    return parser
if __name__ == '__main__':
    args = get_parser().parse_args()
    #model_name = 'MTCNN'
    net = "ONet"
    model_dir = "/content/MTCNN-Tensorflow/data/MTCNN_model/"
    model_path = model_dir + args.model_name + "/%s"%net
    base_dir = args.tfrecord_path
    prefix = model_path
    end_epoch = 1000
    display = 10
    lr = 0.00001
    train_ONet(base_dir, prefix, end_epoch, display, lr)
