#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:52:59 2022

@author: anilbayramg
"""
import cv2
import numpy as np
from old_dehaze import dehaze
from old_dehaze2 import dehaze2
import time

video_path = "/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/demo-inp/short.mp4"

cap = cv2.VideoCapture(video_path)
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/demo-out/short-trial-pure.avi",cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH,HEIGHT))

print("Video has been opened.")
i = 0
#%%
duration = 0
pre_pro = 0
start = 0
post_pro = 0
inference = 0
while cap.isOpened():
    start += time.time()
    (grabbed, np_frame) = cap.read()
    if not grabbed:
        break
#    np_frame = (np_frame - np_frame.min())/(np_frame.max()-np_frame.min()) # Check necessity
#    np_frame = np.interp(np_frame, (np_frame.min(), np_frame.max()),(0,255))
#    np_frame = np_frame.astype(np.uint8)
    output = dehaze(np_frame)
#    output = dehaze2(np_frame)


    output = np.interp(output, (output.min(), output.max()),(0,255))   
    output = cv2.resize(output.astype(np.uint8), (WIDTH,HEIGHT))
    out.write(output)
    post_pro += time.time()
    i += 1
    if i % 30 == 0:
#        pre_pro_avg = (pre_pro - start)/30
#        inf_avg = (inference-pre_pro)/30
#        post_pro_avg = (post_pro - inference)/30
        duration = (post_pro - start)/30
        print(f"{i} frames has been processed.")
        print(f"Average process time: {duration}")
#        print(f"Average pre-processing time: {pre_pro_avg}")
#        print(f"Average inference time: {inf_avg}")
#        print(f"Average post-processing time: {post_pro_avg}")
        print()
        duration = 0
        pre_pro = 0
        start = 0
        post_pro = 0
        inference = 0
cap.release()