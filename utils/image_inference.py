#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:43:03 2022

@author: anilbayramg
"""
import torch

#import numpy as np
import time
#from torchvision import datasets, models, transforms
from torchvision import transforms
from PIL import Image
import dehaze22  as net
#import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Choose an image to pass through the model
test_image = '/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/result_cvpr18/my_image/foggyHouse.jpg'
video_path = "/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/mete-vid.mp4"
#%%

# First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
data_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
# Load the image
image = Image.open(test_image)
plt.figure()
plt.imshow(image)
# Now apply the transformation, expand the batch dimension, and send the image to the GPU
image = data_transform(image).unsqueeze(0).cuda() # For GPU
#image = data_transform(image).unsqueeze(0) # For CPU
image1 = (image - image.min())/ (image.max() - image.min())




# Download the model if it's not there already. It will take a bit on the first run, after that it's fast
netG = net.dehaze(3, 3, 64)
#
#netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cuda')))
netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cuda')))
# model = models.resnet50(pretrained=True)
# Send the model to the GPU 
netG.cuda()
# Set layers such as dropout and batchnorm in evaluation mode
#netG.eval()
#%%

start = time.time()
x_hat, tran_hat, atp_hat, dehaze2= netG(image)
print(str((time.time() - start) )+ " seconds")

start = time.time()
x_hat1, tran_hat, atp_hat, dehaze2= netG(image1)
print(str((time.time() - start) )+ " seconds")
with torch.no_grad():
    plt.figure()
    
    plt.imshow(x_hat[0].cpu().permute(1,2,0))
    plt.figure()
    plt.imshow(x_hat1[0].cpu().permute(1,2,0))
    
#vutils.save_image(x_hat[0], './result_cvpr18/image/real_dehazed/880_my.png', normalize=True, scale_each=False)
    
    

#%%

import cv2
video_path = "/home/anilbayramg/Desktop/Github-desktop/my-DCPDN/mete-vid.mp4"
vs = cv2.VideoCapture(video_path)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (512,512))

grabbed = True
while grabbed:
    (grabbed, frame) = vs.read()
    out , _, _, _ = netG(frame)
    out.write(frame)


    





