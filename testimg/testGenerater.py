import re, os, cv2, sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from datatools import norm
from flood import floodFillDepth

if __name__ == '__main__':
    folders = ['easy', 'complex']
    search = re.compile(r'(\d*)_..*')
    i = 0
    for folder in folders:
        for name in filter(lambda name: search.match(name), os.listdir(folder)):
            img = np.loadtxt(os.path.join(folder, name))
            mask = floodFillDepth(img, (48,48), 10)
            img = img.astype(np.float) / 4096.0
            img = img * mask
            img = cv2.dilate(img, np.ones((3,3)))
            img = cv2.resize(img, (48,48))
            img = norm(img)
            plt.imsave('Test_%d.png' % i, img, format='png', cmap=plt.cm.gray)
            i = i + 1
