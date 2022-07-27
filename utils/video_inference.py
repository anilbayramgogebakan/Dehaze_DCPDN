#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:50:37 2022

@author: anilbayramg
"""

import torch

import numpy as np
#import time
#from torchvision import datasets, models, transforms
from torchvision import transforms
#from PIL import Image
import dehaze22  as net
#import torchvision.utils as vutils
#import matplotlib.pyplot as plt
import cv2



# Choose an image to pass through the model
test_image = '/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/result_cvpr18/my_image/foggyHouse.jpg'
video_path = "/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/demo-inp/very-short.mp4"

data_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

netG = net.dehaze(3, 3, 64)
netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cuda')))
netG.cuda()

transform = transforms.ToTensor()
print("Model has been loaded.")
cap = cv2.VideoCapture(video_path)
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/demo-out/bayram.avi",cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH,HEIGHT))

print("Video has been opened.")
i = 0
#%%
duration = 0
pre_pro = 0
start = 0
post_pro = 0
inference = 0
while cap.isOpened():
#    torch.cuda.synchronize() 
#    start += time.time()
    (grabbed, np_frame) = cap.read()
    if not grabbed:
        break
    np_frame = cv2.resize(np_frame, (512,512), interpolation = cv2.INTER_CUBIC).transpose()
    np_frame = (np_frame - np_frame.min())/(np_frame.max()-np_frame.min())
    frame = torch.tensor(np_frame, dtype=torch.float32)
#    frame = transform(np_frame)
    frame = frame.unsqueeze(0).cuda() # For GPU
#    torch.cuda.synchronize() 
#    pre_pro += time.time()
    
    out_frame , _, _, _ = netG(frame)
#    torch.cuda.synchronize() 
#    inference += time.time()
    
    output = out_frame[0].cpu().detach().numpy().transpose()
    output = np.interp(output, (output.min(), output.max()),(0,255))   
    output = cv2.resize(output.astype(np.uint8), (WIDTH,HEIGHT))
    out.write(output)
#    torch.cuda.synchronize() 
#    post_pro += time.time()
    i += 1
    if i % FPS == 0:
        print("lez go")
#        pre_pro_avg = (pre_pro - start)/30
#        inf_avg = (inference-pre_pro)/30
#        post_pro_avg = (post_pro - inference)/30
#        duration = (post_pro - start)/30
#        print(f"{i} frames has been processed.")
#        print(f"Average process time: {duration}")
#        print(f"Average pre-processing time: {pre_pro_avg}")
#        print(f"Average inference time: {inf_avg}")
#        print(f"Average post-processing time: {post_pro_avg}")
#        print()
#        duration = 0
#        pre_pro = 0
#        start = 0
#        post_pro = 0
#        inference = 0
cap.release()

