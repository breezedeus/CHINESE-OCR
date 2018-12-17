# coding:utf-8
import os
import time
from glob import glob

import numpy as np
import cv2
from PIL import Image

import model
# ces


def parse_image(img_file):
    # img_file = "./test/test.png"
    # img_file = '/Users/king/Documents/WhatIHaveDone/Test/text-detection-ctpn/data/demo/581543991702_.pic_hd.jpg'
    im = Image.open(img_file)
    img = np.array(im.convert('RGB'))
    t = time.time()
    '''
    result,img,angel分别对应-识别结果，图像的数组，文字旋转角度
    '''
    result, boxed_img, angle = model.model(
        img, model='keras', adjust=True, detectAngle=True)
    print("It takes time:{}s".format(time.time() - t))
    return result, boxed_img
    # print("---------------------------------------")
    # for key in result:
    #     print(result[key][1])


if __name__ == '__main__':
    data_dir = 'ctpn/data/demo'
    paths = glob(os.path.join(data_dir, '*.*'))
    for fp in paths:
        if os.path.basename(fp).split('.')[-1] not in {'jpg', 'jpeg', 'png'}:
            continue
        print('parsing file %s ...' % fp)
        result, boxed_im = parse_image(fp)
        res_fp = '.'.join(os.path.basename(fp).split('.')[:-1]) + '_res.txt'
        res_fp = os.path.join(data_dir, res_fp)
        with open(res_fp, 'w') as out_f:
            out_f.writelines(str(result))

        boxed_im_fp = '.'.join(os.path.basename(fp).split('.')[:-1]) + '_boxed.png'
        boxed_im_fp = os.path.join(data_dir, boxed_im_fp)
        cv2.imwrite(boxed_im_fp, boxed_im)
