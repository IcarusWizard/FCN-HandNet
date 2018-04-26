import re, os, cv2
import matplotlib.pyplot as plt
import numpy as np

def Flood(img, l, u):
    flooded = img.copy()
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flooded, mask, (h // 2, w // 2), 1, l, u)
    mask = mask[1:h+1, 1:w+1]
    return img * mask, mask

if __name__ == '__main__':
    l = os.listdir()
    search = re.compile(r'(\d*)_..*')
    for name in filter(lambda name: search.match(name), l):
        print(name)
        num = search.findall(name)[0]
        img = np.loadtxt(name)
        img = img.astype(np.float32) / 4096.0
        #img = plt.imread(name)
        print(img, img.shape, img.dtype)
        plt.subplot(1,3,1)
        plt.imshow(img, cmap=plt.cm.gray)
        img, mask = Flood(img, 15 / 4096.0, 15 / 4096.0)
        plt.subplot(1,3,2)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.subplot(1,3,3)
        plt.imshow(mask, cmap=plt.cm.gray)        
        plt.savefig('Flooded_' + num + '.png')