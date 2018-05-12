import os, re
import numpy as np
import matplotlib.pyplot as plt
from datatools import loadFromH5, smoothFilter
from matplotlib.font_manager import FontProperties
from matplotlib import rc

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')
font.set_size(20)

font_title = FontProperties()
font_title.set_family('serif')
font_title.set_name('Times New Roman')
font_title.set_style('italic')
font_title.set_size(25)

def findStep(modelpath):
    if not os.path.exists(modelpath) or len(os.listdir(modelpath)) == 0:
        print('model path not exist, train with initial point')
        return 0
    else:
        l = os.listdir(modelpath)
        step = 0
        match = re.compile(r'ckpt-(\d*)')
        for name in l:
            result = match.findall(name)
            if not len(result) == 0:
                step = max(step, int(result[0]))
        return step

def drawResult(img, stage):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.subplot(3,4,3)
    plt.imshow(stage[:,:,0])
    plt.subplot(3,4,4)
    plt.imshow(stage[:,:,1])
    plt.subplot(3,4,7)
    plt.imshow(stage[:,:,2])
    plt.subplot(3,4,8)
    plt.imshow(stage[:,:,3])
    plt.subplot(3,4,11)
    plt.imshow(stage[:,:,4])
    plt.subplot(3,4,12)
    plt.imshow(stage[:,:,5]) 

def drawMerit(*, filename = 'merit.h5', show = True):
    step, loss, acc = loadFromH5(filename, ['step', 'loss', 'acc'])
    if show:
        plt.figure()
        for i in range(3):
            plt.plot(step, smoothFilter(loss[:,i]))
        plt.figure()
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            for i in range(3):
                plt.plot(step, smoothFilter(acc[:,i*3+j]))
        plt.show()
    else:
        if not os.path.exists('merit'):
            os.mkdir('merit')
        names = ['Thumb', 'Index', 'Middle', 'Ring', 'Small', 'Palm']
        colors = ['b', 'g', 'r']
        fig, ax = plt.subplots(constrained_layout=True)
        for i in range(3):
            ax.plot(step, smoothFilter(loss[:,i]), colors[i])
        ax.set_xlabel('Train Step', fontproperties=font)
        ax.set_ylabel('MSE Loss', fontproperties=font)
        ax.set_title('Loss', fontproperties=font_title)
        ax.axis([0, 300000, 4, 18])
        axis = fig.axes[0]
        axis.tick_params(labelsize=15)
        ax.grid()
        fig.savefig('merit/' + 'Loss.png')
        for j in range(6):
            fig, ax = plt.subplots(constrained_layout=True)
            for i in range(3):
                ax.plot(step, smoothFilter(acc[:,i*3+j]), colors[i])
            ax.set_xlabel('Train Step', fontproperties=font)
            ax.set_ylabel('Acc', fontproperties=font)
            ax.set_title(names[j], fontproperties=font_title)
            ax.axis([0, 300000, 0, 1])
            axis = fig.axes[0]
            axis.tick_params(labelsize=15)
            ax.grid()
            fig.savefig('merit/' + names[j] + '.png')
        

def drawFeatures(fetures):
    size = fetures.shape[2]
    base = int(np.log2(size))
    rows = 2 ** (base // 2)
    cols = 2 ** (base // 2 + base % 2)
    plt.figure()
    for i in range(size):
        plt.subplot(rows, cols, i+1)
        plt.imshow(fetures[:,:,i])

if __name__ == '__main__':
    drawMerit(show=False)