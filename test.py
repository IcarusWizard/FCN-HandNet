from datatools import batch_data
from model import FCN_Handnet
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

if __name__ == '__main__':
    test_data_path = '.\\TestData'

    sess = tf.InteractiveSession()

    train_op, img, label, stage, loss, acc = FCN_Handnet()

    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-52470')

    m = 0
    loss_total = [0, 0, 0]
    acc_total = [np.zeros(6), np.zeros(6), np.zeros(6)]
    for img_batch, label_batch in batch_data(test_data_path, 32):
        m += img_batch.shape[0]
        loss_val, acc_val = sess.run([loss, acc], feed_dict={img : img_batch, label : label_batch})
        loss_total = [loss_total[i] + loss_val[i] for i in range(3)]
        acc_total = [acc_total[i] + acc_val[i] for i in range(3)]

    loss_total = [loss_total[i] / m for i in range(3)]
    acc_total = [acc_total[i] / m for i in range(3)]
    print("In %d Test:" % m)
    print("loss = ", loss_total)
    print("acc = ", acc_total)

    for img_batch, label_batch in batch_data(test_data_path, 1):
        stage_val = sess.run(stage, feed_dict={img : img_batch, label : label_batch})
        img_batch = img_batch[0,:,:,0]
        label_batch = label_batch[0]
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_batch, cmap=plt.cm.gray)
        plt.subplot(2,6,4)
        plt.imshow(label_batch[:,:,0])
        plt.subplot(2,6,5)
        plt.imshow(label_batch[:,:,1])
        plt.subplot(2,6,6)
        plt.imshow(label_batch[:,:,2])
        plt.subplot(2,6,10)
        plt.imshow(label_batch[:,:,3])
        plt.subplot(2,6,11)
        plt.imshow(label_batch[:,:,4])
        plt.subplot(2,6,12)
        plt.imshow(label_batch[:,:,5])        
        for _stage in stage_val:
            _stage = _stage[0]
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(img_batch, cmap=plt.cm.gray)
            plt.subplot(2,6,4)
            plt.imshow(_stage[:,:,0])
            plt.subplot(2,6,5)
            plt.imshow(_stage[:,:,1])
            plt.subplot(2,6,6)
            plt.imshow(_stage[:,:,2])
            plt.subplot(2,6,10)
            plt.imshow(_stage[:,:,3])
            plt.subplot(2,6,11)
            plt.imshow(_stage[:,:,4])
            plt.subplot(2,6,12)
            plt.imshow(_stage[:,:,5])
        plt.show()
