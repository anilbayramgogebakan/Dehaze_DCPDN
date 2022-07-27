#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:36:52 2022

@author: anilbayramg
"""
import cv2
import os


im_arr = os.listdir("./Yunus-bey-demo")


for im in im_arr:
    image = cv2.imread("./Yunus-bey-demo/" + im)
    
    print("Heıght: " + str(image.shape[0]))
    print("Width: " + str(image.shape[1]))
    
    H1 = int(input("Enter desired height-1:"))
    H2 = int(input("Enter desired height-2:"))
    W1 = int(input("Enter desired width-1:"))
    W2 = int(input("Enter desired width-2:"))
    
    cropped_image = image[H1:H2,W1:W2]
    
    cv2.imwrite("crop"+im, cropped_image)