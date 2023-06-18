import os
from os import listdir, getcwd
import os.path as osp
import shutil
import natsort
import numpy as np


source_f = './fault/'
source_n = './normal/'

train_f = '../dataset/train/fault/'
train_n = '../dataset/train/normal/'

test_f = '../dataset/test/fault/'
test_n = '../dataset/test/normal/'


#train : test = 80 : 20 비율로 랜덤선택하여 복사한다.
test_size = 0.2

##### fault data --> train, test split
try:
    list_dir = os.listdir(source_f)
    list_dir = natsort.natsorted(list_dir, reverse=False)
    imlist = [osp.join(osp.realpath('.'), source_f, img) for img in list_dir if os.path.splitext(img)[1] =='.jpg'  or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] =='.JPG' or os.path.splitext(img)[1] =='.png']
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), source_f))
    print('Not a directory error')
except FileNotFoundError:
    print ("No file or directory with the name {}".format(source_f))
    exit()

num_fault = len(imlist)
indices = list(range(num_fault))
np.random.shuffle(indices)
split = int(np.floor(test_size * num_fault))
train_idx, test_idx = indices[split:], indices[:split]

for inx in train_idx:
    print('\n'+ list_dir[inx])
    shutil.copyfile(source_f + list_dir[inx], train_f + list_dir[inx] )
    
for inx in test_idx:
    print('\n'+ list_dir[inx])
    shutil.copyfile(source_f + list_dir[inx], test_f + list_dir[inx] )


#### normal data --> train, test split
try:
    list_dir = os.listdir(source_n)
    list_dir = natsort.natsorted(list_dir, reverse=False)
    imlist = [osp.join(osp.realpath('.'), source_n, img) for img in list_dir if os.path.splitext(img)[1] =='.jpg'  or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] =='.JPG' or os.path.splitext(img)[1] =='.png']
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), source_n))
    print('Not a directory error')
except FileNotFoundError:
    print ("No file or directory with the name {}".format(source_n))
    exit()

num_fault = len(imlist)
indices = list(range(num_fault))
np.random.shuffle(indices)
split = int(np.floor(test_size * num_fault))
train_idx, test_idx = indices[split:], indices[:split]

for inx in train_idx:
    print('\n'+ list_dir[inx])
    shutil.copyfile(source_n + list_dir[inx], train_n + list_dir[inx] )
    
for inx in test_idx:
    print('\n'+ list_dir[inx])
    shutil.copyfile(source_n + list_dir[inx], test_n + list_dir[inx] )
    
