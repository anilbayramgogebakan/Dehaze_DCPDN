#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:59:40 2022

@author: anilbayramg
"""
import os 

path = "/home/anilbayramg/Desktop/Github/DCPDN/result_cvpr18/my_prepro"


for i, name in enumerate(os.listdir(path), 0):
    
    dst = "{}/{}.h5".format(path, i)
    src = "{}/{}".format(path, name)
    os.rename(src, dst)
    
