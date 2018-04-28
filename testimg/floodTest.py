import re, os, cv2
import matplotlib.pyplot as plt
import numpy as np
from flood import floodFillDepth

if __name__ == '__main__':
    floder = os.sys.argv[1]
    l = os.listdir(floder)
    search = re.compile(r'(\d*)_..*')
    for name in filter(lambda name: search.match(name), l):
        print(name)
        num = search.findall(name)[0]
        img = np.loadtxt(os.path.join(floder, name))
        mask = floodFillDepth(img, (48,48), 10)
        img = img.astype(np.float32) / 4096.0
        plt.subplot(1,3,1)
        plt.imshow(img, cmap=plt.cm.gray)
        img = img * mask
        img = cv2.dilate(img, np.ones((3,3))) #dilate to make the image more fat
        plt.subplot(1,3,2)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.subplot(1,3,3)
        plt.imshow(mask, cmap=plt.cm.gray)        
        plt.savefig('Flooded_' + num + '.png')