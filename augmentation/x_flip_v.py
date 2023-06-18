import os
from os import listdir, getcwd
import os.path as osp
import cv2
import natsort


#test = './png_f'
#target = './png_f_fv'
test = './png_n'
target = './png_n_fv'
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

   #수평 뒤집기
    transformed_image = cv2.flip(image, 0) # 0:vertical

    dirname, basename = os.path.split(img)
    name, ext = os.path.splitext(basename)
    cv2.imwrite(target + '/' + name + '_fv' + ext, transformed_image)
   # cv2.imshow('image',transformed_image) 
   # cv2.waitKey(0) # 키보드 입력을 대기하는 함수, milisecond값을 넣으면 해당 시간동안 대기, 0인경우 무한으로 대기
   # cv2.destroyAllWindows()    