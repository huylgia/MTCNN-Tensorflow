#coding:utf-8
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from train_models.MTCNN_config import config

sys.path.append("../prepare_data")
print(sys.path)
from prepare_data.read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord

import random
import cv2
def train_model(base_lr, loss, data_num,end_epoch):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    global_step = tf.Variable(0, trainable=False)
    #Origin code
    '''LR_EPOCH [8,14]
    lr_factor = 0.1
    boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]'''
    #Adjust code
    epoch_range = [int(0.05*end_epoch*i) for i in range(1,21)]
    boundaries = [epoch * data_num // config.BATCH_SIZE if data_num % config.BATCH_SIZE == 0 else epoch * (data_num // config.BATCH_SIZE + 1)
                  for epoch in epoch_range]
    print(boundaries)
    end_lr = 0.0000009
    lr_values = [base_lr - (base_lr - end_lr)*epoch/len(epoch_range) for epoch 
    in range(0, len(epoch_range) + 1)]

    #control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op,0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op

'''
certain samples mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    num_images = image_batch.shape[0]
    random_number = npr.choice([0,1],num_images,replace=True)
    #the index of image needed to flip
    indexes = np.where(random_number>0)[0]
    fliplandmarkindexes = np.where(label_batch[indexes]==-2)[0]
    
    #random flip    
    for i in indexes:
        cv2.flip(image_batch[i],1,image_batch[i])
    #pay attention: flip landmark    
    for i in fliplandmarkindexes:
        landmark_ = landmark_batch[i].reshape((-1,2))
        landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
        landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
        landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
        landmark_batch[i] = landmark_.ravel()
    return image_batch,landmark_batch
'''
# all mini-batch mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    #mirror
    if random.choice([0,1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch==-2)[0]
        flipposindexes = np.where(label_batch==1)[0]
        #only flip
        flipindexes = np.concatenate((fliplandmarkindexes,flipposindexes))
        #random flip    
        for i in flipindexes:
            cv2.flip(image_batch[i],1,image_batch[i])        
        
        #pay attention: flip landmark    
        for _,i in enumerate(fliplandmarkindexes):
            landmark_ = landmark_batch[i].reshape((-1,2))
            landmark_ = np.asarray([(x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
            landmark_[[2, 3]] = landmark_[[3, 2]]#left mouth<->right mouth    
            landmark_batch[i] = landmark_.ravel()
        
    return image_batch,landmark_batch

def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)

    return inputs

def train(net_factory, prefix, end_epoch, base_dir,
          display=200, base_lr=0.01, pretrained_model=None):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param prefix: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    print('đm prefix: ', prefix)
    net = prefix.split('/')[-1]
    #label file
    label_file = os.path.join(base_dir,'train_%s_landmark.txt' % net)
    #label_file = os.path.join(base_dir,'landmark_12_few.txt')
    # print(label_file)
    # f = open(label_file, 'r')
    # # get number of training examples
    # num = len(f.readlines())
    num = config.NUM
    # print("Total size of the dataset is: ", num)
    # print(prefix)

    #PNet use this method to get data
    if net == 'PNet':
        #dataset_dir = os.path.join(base_dir,'train_%s_ALL.tfrecord_shuffle' % net)
        dataset_dir = os.path.join(base_dir,'train_%s_landmark.tfrecord_shuffle' % net)
        print('dataset dir is:',dataset_dir)
        image_batch, label_batch, bbox_batch,landmark_batch = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, net)
        print('đúng mà te: ', image_batch.get_shape())
        
    #RNet use 3 tfrecords to get data    
    else:
        plate_dir = os.path.join(base_dir,'')
        #landmark_dir = os.path.join(base_dir,'landmark_landmark.tfrecord_shuffle')
        # landmark_dir = os.path.join('../../DATA/imglists/RNet','landmark_landmark.tfrecord_shuffle')
        dataset_dirs = plate_dir
        # landmark_radio=1.0/6
        # landmark_batch_size = int(np.ceil(config.BATCH_SIZE*landmark_radio))
        landmark_batch_size = config.BATCH_SIZE
        assert landmark_batch_size != 0,"Batch Size Error "
        batch_sizes = landmark_batch_size
        #print('batch_size is:', batch_sizes)
        image_batch, label_batch, bbox_batch,landmark_batch = read_multi_tfrecords(dataset_dirs,batch_sizes, net)
        print('tooturu: ', image_batch.get_shape())        
        
    #landmark_dir    
    if net == 'PNet':
        image_size = 12
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    elif net == 'RNet':
        image_size = 24
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 0.5;
    else:
        radio_cls_loss = 1.0;radio_bbox_loss = 0.5;radio_landmark_loss = 1;
        image_size = 48
    
    #define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32,shape=[config.BATCH_SIZE,8],name='landmark_target')
    #get loss and accuracy
    input_image = image_color_distort(input_image)
    cls_loss_op,bbox_loss_op,landmark_loss_op,L2_loss_op,accuracy_op = net_factory(input_image, label, bbox_target,landmark_target,training=True)
    #train,update learning rate(3 loss)
    total_loss_op  = radio_cls_loss*cls_loss_op + radio_bbox_loss*bbox_loss_op + radio_landmark_loss*landmark_loss_op + L2_loss_op
    train_op, lr_op = train_model(base_lr,
                                  total_loss_op,
                                  num,
                                  end_epoch)
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()


    #save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    #pretrained_model
    if pretrained_model:
      print('Restoring pretrained model: %s' % pretrained_model)
      ckpt = tf.train.get_checkpoint_state(pretrained_model)
      print(ckpt)
      saver.restore(sess, ckpt.model_checkpoint_path)
      
    #visualize some variables
    tf.summary.scalar("cls_loss",cls_loss_op)#cls_loss
    tf.summary.scalar("bbox_loss",bbox_loss_op)#bbox_loss
    tf.summary.scalar("landmark_loss",landmark_loss_op)#landmark_loss
    tf.summary.scalar("cls_accuracy",accuracy_op)#cls_acc
    tf.summary.scalar("total_loss",total_loss_op)#cls_loss, bbox loss, landmark loss and L2 loss add together
    summary_op = tf.summary.merge_all()
    logs_dir = "logs/%s" %(net)
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer,projector_config)
    #begin 
    coord = tf.train.Coordinator()
    #begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    #total steps
    '''Adjust'''
    if num % config.BATCH_SIZE != 0:
        MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    else:
        MAX_STEP = int(num / config.BATCH_SIZE ) * end_epoch
    epoch = 0
    sess.graph.finalize()
    try:
        #Origin code
        '''for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            print(image_batch, label_batch, bbox_batch,landmark_batch)
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            #random flip
            #image_batch_array,landmark_batch_array = random_flip_images(image_batch_array,label_batch_array,landmark_batch_array)
            # print('im here')
            # print(image_batch_array.shape)
            # print(label_batch_array.shape)
            # print(bbox_batch_array.shape)
            # print(landmark_batch_array.shape)
            # print(label_batch_array[0])
            # print(bbox_batch_array[0])
            # print(landmark_batch_array[0])

            # print('im here')
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            if (step+1) % display == 0:
                #acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

                total_loss = radio_cls_loss*cls_loss + radio_bbox_loss*bbox_loss + radio_landmark_loss*landmark_loss + L2_loss
                # landmark loss: %4f,
                print("%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                datetime.now(), step+1,MAX_STEP, acc, cls_loss, bbox_loss,landmark_loss, L2_loss,total_loss, lr))


            #save every two epochs
            if i * config.BATCH_SIZE >= num:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, prefix, global_step=epoch)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary,global_step=step)'''
        #Adjust code
        minimize_loss = 1e+99
        metrics = []
        avg = lambda items: float(sum(items)) / len(items)
        model_dir = os.path.dirname(prefix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log_file = model_dir + "/log.csv"
        f = open(log_file,"w")
        info = "" 
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array,landmark_batch_array = sess.run([image_batch, label_batch, bbox_batch,landmark_batch])
            _,_,summary = sess.run([train_op, lr_op ,summary_op], feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,landmark_target:landmark_batch_array})
            cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc = sess.run([cls_loss_op, bbox_loss_op,landmark_loss_op,L2_loss_op,lr_op,accuracy_op],
                                                             feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})
            metrics.append([cls_loss, bbox_loss,landmark_loss,L2_loss,lr,acc])

            if i * config.BATCH_SIZE >= num:
                epoch = epoch + 1
                metrics_tranpose = zip(*metrics)
                avg_cls_loss, avg_bbox_loss,avg_landmark_loss,avg_L2_loss,avg_lr,avg_acc = map(avg,metrics_tranpose)
                total_loss = radio_cls_loss*avg_cls_loss + radio_bbox_loss*avg_bbox_loss + radio_landmark_loss*avg_landmark_loss + avg_L2_loss
                string = "%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                datetime.now(), step+1,MAX_STEP, avg_acc, avg_cls_loss, avg_bbox_loss,avg_landmark_loss, avg_L2_loss,total_loss, avg_lr)
                print(string)
                if total_loss < minimize_loss:
                    minimize_loss = total_loss
                    path_prefix = saver.save(sess, prefix, global_step=epoch)
                    print('path prefix is :', path_prefix)
                    info += string + "\n"
                i = 0
                metrics = []
            writer.add_summary(summary,global_step=step)
        f.write(info[:-1])
        f.close()
    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()
