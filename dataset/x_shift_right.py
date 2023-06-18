import os
from os import listdir, getcwd
import os.path as osp
import glob
import torchvision
from torchvision import transforms 
import torch.backends.cudnn as cudnn

import cv2
import natsort
import albumentations as A
import numpy as np


#test = './png_f'
#target = './png_f_sr'
test = './png_n'
target = './png_n_sr'
try:
    list_dir = os.listdir(test)
    list_dir = natsort.natsorted(list_dir, reverse=False)
    imlist = [osp.join(osp.realpath('.'), test, img) for img in list_dir if os.path.splitext(img)[1] =='.jpg'  or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] =='.JPG' or os.path.splitext(img)[1] =='.png']
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), test))
    print('Not a directory error')
except FileNotFoundError:
    print ("No file or directory with the name {}".format(test))
    exit()

for inx, img in enumerate(imlist):

    print('\n'+ list_dir[inx])
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    shift = 100
    #오른쪽으로 시프트
    M = np.float32([[1,0, shift], [0,1,0]])
    len = image.shape[1]
    len2 = image.shape[0]
    transformed_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    transformed_image[:, 0:shift] = image[ :, 0:shift]
    dirname, basename = os.path.split(img)
    name, ext = os.path.splitext(basename)
    cv2.imwrite(target + '/' + name + '_sr' + ext, transformed_image)
   # cv2.imshow('image',transformed_image) 
   # cv2.waitKey(0) # 키보드 입력을 대기하는 함수, milisecond값을 넣으면 해당 시간동안 대기, 0인경우 무한으로 대기
   # cv2.destroyAllWindows()    