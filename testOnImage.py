from datatools import norm
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os, cv2
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

if __name__ == '__main__':
    testimg = plt.imread(os.sys.argv[1])
    testimg = cv2.resize(testimg, (48,48))
    testimg[40:48,10:18] = 0
    #testimg = cv2.dilate(testimg, np.ones((3,3)), iterations = 1)
    testimg = testimg[:,-1::-1]
    testimg = norm(testimg.reshape((48,48,1)))
    testimg = testimg[np.newaxis]

    sess = tf.InteractiveSession()

    train_op, img, label, features, stage, loss, acc = FCN_Handnet()

    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-' + str(findStep('Model')))
    features_val, stage_val = sess.run([features, stage], feed_dict={img : testimg})
    testimg = testimg[0,:,:,0]
    drawFeatures(features_val[0])    
    for _stage in stage_val:
        _stage = _stage[0]
        drawResult(testimg, _stage)
    plt.show()