import h5py, os, re, cv2
os.sys.path.append('testimg/')
from flood import floodFillDepth
import matplotlib.pyplot as plt
import numpy as np
from datatools import buileToH5, norm

def Rot(img, M, size=(48,48)):
    rotated = norm(cv2.warpAffine(img, M, size))
    return rotated[:,:,np.newaxis]

search = re.compile(r'(\d*)_..*')
floders = ['testimg/easy', 'testimg/complex']

whole_hand = []
every_finger = []
cot = 0

for folder in floders:
    for name in filter(lambda name: search.match(name), os.listdir(folder)):
        index = int(search.findall(name)[0])
        img = np.loadtxt(os.path.join(folder, name))
        labels = np.loadtxt(os.path.join(folder, "label%d.txt" % index))
        img = floodFillDepth(img, (48, 48), 20) * img
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_NEAREST)
        fingerlist = []
        for i in range(6):
            x, y = labels[i]
            x = int(x) // 2
            y = int(y) // 2
            tem = np.zeros((48,48))
            tem[x, y] = 1
            tem = cv2.GaussianBlur(tem, (3,3), 0)
            fingerlist.append(tem)
        for angle in np.linspace(-180, 180, 37):
            M = cv2.getRotationMatrix2D((24,24), angle, 1)
            rotated = Rot(img, M)
            whole_hand.append(rotated[np.newaxis])
            rotated = np.concatenate([Rot(x, M) for x in fingerlist], axis=2)
            every_finger.append(rotated[np.newaxis])
            cot = cot + 1
whole_hand = np.concatenate(whole_hand, axis=0)
every_finger = np.concatenate(every_finger, axis=0)
if not os.path.exists('fineturnData'):
    os.mkdir('fineturnData')
filename = "fineturnData/fineturn.h5"
dicts = {'num' : cot, 'whole_hand' : whole_hand, 'every_finger' : every_finger}
buileToH5(filename, dicts, False)
print('finish ' + filename + ' with %d samples' %cot)