import os
import re
import matplotlib.pyplot as plt

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
    plt.subplot(2,6,4)
    plt.imshow(stage[:,:,0])
    plt.subplot(2,6,5)
    plt.imshow(stage[:,:,1])
    plt.subplot(2,6,6)
    plt.imshow(stage[:,:,2])
    plt.subplot(2,6,10)
    plt.imshow(stage[:,:,3])
    plt.subplot(2,6,11)
    plt.imshow(stage[:,:,4])
    plt.subplot(2,6,12)
    plt.imshow(stage[:,:,5]) 