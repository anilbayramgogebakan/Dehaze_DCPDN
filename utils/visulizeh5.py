# import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
import os

def show_im(key):
    plt.figure()
    plt.imshow(key)

def show(index, input_path):
    f = h5py.File('{}/{}.h5'.format(input_path, index), 'r')
    result_path = "/home/anilbayramg/Desktop/Github/DCPDN/result_cvpr18/image/real_dehazed/{}_DCPCN.png".format(index)
    res = plt.imread(result_path)
    
    haze_image=f['haze'][:]
    print(haze_image.shape)
    gt_trans_map=f['trans'][:]
    gt_ato_map=f['ato'][:]
    GT=f['gt'][:]
    
    plt.figure(index)
    
    plt.subplot(121)
    plt.imshow(haze_image)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(res)
    plt.axis('off')
    
    plt.savefig("/home/anilbayramg/Desktop/Github/DCPDN/comparision/{}_comp.png".format(index))
    
    
    

# input_path = "/home/anilbayramg/Desktop/Github/DCPDN/facades/val512"
input_path = "/home/anilbayramg/Desktop/Github/DCPDN/result_cvpr18/my_prepro"
Ex_num = len(os.listdir(input_path))-1
rand_ind = random.sample(range(0, len(os.listdir(input_path))-1), Ex_num)

for ind in rand_ind:
    show(ind, input_path)

# plt.close('all')
