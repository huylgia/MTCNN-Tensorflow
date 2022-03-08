#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 6
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [220000,420000,430000]
