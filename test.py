from datatools import batch_data
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

if __name__ == '__main__':
    train_data_path = 'TrainData'
    validation_data_path = 'ValidationData'
    test_data_path = 'TestData'

    sess = tf.InteractiveSession()

    train_op, img, label, features, stage, loss, acc, correct = FCN_Handnet(acc_range=3)

    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-' + str(findStep('Model')))
    #drawMerit()

    def eval_dataset(dataset):
        m = 0
        loss_total = 0
        acc_total = np.zeros(6)
        acc_whole = 0
        acc_estimate = 0
        for img_batch, label_batch in batch_data(dataset, 16):
            m += img_batch.shape[0]
            loss_val, acc_val, correct_val = sess.run([loss[2], acc[2], correct[2]], feed_dict={img : img_batch, label : label_batch})
            loss_total += loss_val
            acc_total += acc_val
            acc_whole += np.sum(np.all(correct_val[:,0:5], axis=1).astype('float'))
            acc_total += np.sum((correct_val[:,0] & np.any(correct_val[:,1:5], axis=1).astype('float')))
        
        loss_average = loss_total / m
        acc_single_average = acc_total[0:5] / m
        acc_whole_average = acc_whole / m
        acc_estimate_average = acc_estimate / m
        print('--------------------------------------------------')
        print('In DataSet %s:' %dataset)
        print('Average Loss is %f' %loss_average)
        print('Average Single Acc is %.2f' %(np.sum(acc_single_average) / 5))
        print('Average Thumb Single Acc is %.2f' %acc_single_average[0])
        print('Average Index Single Acc is %.2f' %acc_single_average[1])
        print('Average Middle Single Acc is %.2f' %acc_single_average[2])
        print('Average Ring Single Acc is %.2f' %acc_single_average[3])
        print('Average Small Single Acc is %.2f' %acc_single_average[4])
        print('Average Estimate Acc is %.2f' %acc_estimate_average)
        print('Average Whole Hand Acc is %.2f' %acc_whole_average)

    eval_dataset(train_data_path)
    eval_dataset(validation_data_path)
    eval_dataset(test_data_path)

    for img_batch, label_batch in batch_data(test_data_path, 1):
        features_val, stage_val = sess.run([features, stage], feed_dict={img : img_batch, label : label_batch})
        img_batch = img_batch[0,:,:,0]
        label_batch = label_batch[0]
        drawFeatures(features_val[0])
        drawResult(img_batch, label_batch)     
        for _stage in stage_val:
            _stage = _stage[0]
            drawResult(img_batch, _stage)
        plt.show()
