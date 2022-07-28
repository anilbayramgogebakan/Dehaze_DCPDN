#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:08:33 2022

@author: anilbayramg
"""

# import the opencv library
import cv2
import dehaze22  as net
import torch
import numpy as np
import time


netG = net.dehaze(3, 3, 64)

# netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cuda'))) # For GPU
# netG.cuda() # For GPU
netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cpu'))) # For CPU

#%% For FPS calculation

start = 0
finish = 0
i = 0
## For FPS calculation

#%%
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
    torch.cuda.synchronize() 
    start += time.time()
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # cv2.imshow('input', frame) # Comment out to watch input stream
    
    frame = cv2.resize(frame, (512,512), interpolation = cv2.INTER_CUBIC).transpose() # Resize
    frame = (frame - frame.min())/(frame.max()-frame.min()) # Normalize
    frame = torch.tensor(frame, dtype=torch.float32) # Convert to tensor
    
    # frame = frame.unsqueeze(0).cuda() # For GPU
    frame = frame.unsqueeze(0) # For CPU
    
    with torch.no_grad():
        out_frame , _, _, _ = netG(frame)
        show = out_frame[0].cpu().permute(2,1,0)
        
        cv2.imshow('dehazed', np.array(show))
  
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    torch.cuda.synchronize() 
    finish += time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #%% For FPS calculation
    i += 1
    if i % 30 == 0:
        print("FPS: {}".format(1/((finish-start)/30)))
        start = 0
        finish = 0
    
    ## For FPS calculation
    #%%
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()