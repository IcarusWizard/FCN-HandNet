from datatools import norm
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os, cv2, re
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

if __name__ == '__main__':
    folder = 'testimg'
    search = re.compile(r'Test_(\d*)..*')

    sess = tf.InteractiveSession()
    train_op, img, label, features, stage, loss, acc = FCN_Handnet()
    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-' + str(findStep('Model')))
    
    for name in filter(lambda name: search.match(name), os.listdir(folder)):
        testimg = plt.imread(os.path.join(folder, name))
        testimg = testimg[:,:,0]
        #testimg = testimg[:,-1::-1] #swap if need
        testimg = norm(testimg.reshape((48,48,1)))
        testimg = testimg[np.newaxis]
        features_val, stage_val = sess.run([features, stage], feed_dict={img : testimg})
        testimg = testimg[0,:,:,0]
        drawFeatures(features_val[0])    
        for _stage in stage_val:
            _stage = _stage[0]
            drawResult(testimg, _stage)
        plt.show()