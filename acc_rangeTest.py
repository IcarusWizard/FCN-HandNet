from datatools import batch_data, loadFromH5, buileToH5
from model import FCN_Handnet
from modeltools import *
import tensorflow as tf
import tensorlayer as tl
import win_unicode_console, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
win_unicode_console.enable()

font_axis = FontProperties()
font_axis.set_family('serif')
font_axis.set_name('Times New Roman')
font_axis.set_style('italic')
font_axis.set_size(20)

font_title = FontProperties()
font_title.set_family('serif')
font_title.set_name('Times New Roman')
font_title.set_style('italic')
font_title.set_size(25)

def findmax(label, imgsize=48):
    m = label.shape[0]
    label_flat = label.reshape(m, imgsize ** 2)
    index = np.argmax(label_flat, axis=1)[:, np.newaxis]
    return np.concatenate([index // imgsize, index % imgsize], axis=1)

def findAllmax(labels): #m*48*48*6 -> m*2*6
    return np.concatenate([findmax(labels[:,:,:,i])[:,:,np.newaxis] for i in range(6)], axis=2)

if __name__ == '__main__':
    paths = ['ValidationData', 'TestData']
    filename = 'acc_range.h5'
    if not os.path.exists(filename):
        store_dict = dict()

        acc_range_vec = []
        for i in range(10):
            for j in range(i, 10):
                d = np.sqrt(i ** 2 + j ** 2)
                if not d in acc_range_vec:
                    acc_range_vec.append(d)
        acc_range_vec = sorted(acc_range_vec)
        store_dict['acc_range'] = acc_range_vec

        sess = tf.InteractiveSession()

        train_op, img, label, features, stage, loss, acc = FCN_Handnet(acc_range=3)

        saver = tf.train.Saver()
        saver.restore(sess, './Model/model.ckpt-' + str(findStep('Model')))

        for test_data_path in paths:
            stage_vec = [[], [], []]
            label_vec = []
            for img_batch, label_batch in batch_data(test_data_path, 8):
                stage_tem = sess.run(stage, feed_dict={img : img_batch, label : label_batch})
                for i in range(3):
                    stage_vec[i].append(findAllmax(stage_tem[i]))
                label_vec.append(findAllmax(label_batch))

            stage_val = [np.concatenate(stage_vec[i], axis=0) for i in range(3)]
            label_val = np.concatenate(label_vec, axis=0)
            distance = [np.sqrt(np.sum((stage_val[i] - label_val) ** 2, axis=1)) for i in range(3)]
            m = label_val.shape[0]

            acc_vec = [[], [], []]
            for acc_range in acc_range_vec:
                for i in range(3):
                    acc_vec[i].append(np.sum(distance[i] <= acc_range, axis=0)[np.newaxis] / m)

            acc_val = [np.concatenate(acc_vec[i], axis=0)[np.newaxis] for i in range(3)]
            acc_val = np.concatenate(acc_val, axis=0)#3*m*6
            store_dict[test_data_path] = acc_val

        buileToH5(filename, store_dict)
    else:
        output_path = 'AccRange'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        datasets = ['ValidationData', 'TestData']
        names = ['Thumb', 'Index', 'Middle', 'Ring', 'Small', 'Palm']
        colors = ['b', 'g', 'r']
        acc_range, validation, test = loadFromH5(filename, ['acc_range', 'ValidationData', 'TestData'])
        acc = np.concatenate([validation[np.newaxis], test[np.newaxis]], axis=0)
        for i in range(2):
            for j in range(6):
                fig, ax = plt.subplots(constrained_layout=True)
                for k in range(3):
                    ax.plot(acc_range, acc[i, k, :, j], colors[k])
                ax.set_xlabel('Acc Range: R', fontproperties=font_axis)
                ax.set_ylabel('Acc', fontproperties=font_axis)
                ax.set_title(names[j], fontproperties=font_title)
                ax.axis([0, 15, 0, 1])
                axis = fig.axes[0]
                axis.tick_params(labelsize=15)
                ax.grid()
                fig.savefig(os.path.join(output_path, datasets[i], names[j] + '.png'))
