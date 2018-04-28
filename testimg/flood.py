import numpy as np
from queue import Queue

def floodFillDepth(img, startPoint, threshold):
    mask = np.zeros(img.shape, dtype=np.uint8)
    h, w = img.shape
    q = Queue()
    q.put((startPoint, startPoint))
    while not q.empty():
        prepoint, point = q.get()
        x, y = point
        if(x < h and x >= 0 and y < w and y >= 0 and mask[point] == 0 and np.abs(img[prepoint] - img[point]) <= threshold):
            mask[point] = 1
            q.put((point, (x + 1, y)))
            q.put((point, (x - 1, y)))
            q.put((point, (x, y + 1)))
            q.put((point, (x, y - 1)))
    return mask