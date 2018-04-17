from datatools import batch_data
from model import FCN_Handnet
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
import matplotlib.pyplot as plt
win_unicode_console.enable()

if __name__ == '__main__':
    test_data_path = '.\\ValidationData'

    sess = tf.InteractiveSession()

    train_op, img, label, stage, loss, acc = FCN_Handnet()

    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-2620')

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
