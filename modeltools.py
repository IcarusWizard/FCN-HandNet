import os
import re

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