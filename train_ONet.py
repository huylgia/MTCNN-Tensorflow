#coding:utf-8
from train_models.mtcnn_model import O_Net
from train_models.train import train


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

if __name__ == '__main__':
    base_dir = '/content/MTCNN-Tensorflow/plate_landmark.tfrecord_shuffle'

    model_name = 'MTCNN'
    model_path = '/content/MTCNN-Tensorflow/data/%s_model/Yellow_ONet_landmark_25-02-2022/ONet' % model_name
    prefix = model_path
    end_epoch = 20000
    display = 10
    lr = 0.0001
    train_ONet(base_dir, prefix, end_epoch, display, lr)
