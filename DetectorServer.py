from datatools import norm
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os, cv2, re, socket, time
import matplotlib.pyplot as plt
import numpy as np
win_unicode_console.enable()

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    train_op, img, label, features, stage, loss, acc, correct = FCN_Handnet()
    saver = tf.train.Saver()
    saver.restore(sess, './Model/model.ckpt-' + str(findStep('Model')))

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 9999))
    s.listen(5)
    print('Detector server waiting for connection...')

    while True:
        sock, addr = s.accept()
        print('Accpet new connection from %s:%s' % addr)
        starttime = time.time()
        data = sock.recv(102400)
        testimg = norm(np.array(list(map(int, data.split(b' ')))).reshape(48,48,1).astype(np.float32))
        testimg = testimg[np.newaxis]
        stage_val = sess.run(stage[-1], feed_dict={img : testimg})[0]
        result = ""
        for i in range(5):
            _stage = stage_val[:,:,i]
            index = np.argmax(_stage)
            result = result + str(index // 48) + " " + str(index % 48)
            if not i == 4:
                result = result + " "
        sock.send(result.encode('utf-8'))
        print('%.2f ms is costed on server' %((time.time() - starttime) * 1000))
        sock.close()
        print('End connection from %s:%s' % addr)