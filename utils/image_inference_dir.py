#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:43:03 2022

@author: anilbayramg
"""
import torch

# import numpy as np
import time
#from torchvision import datasets, models, transforms
from torchvision import transforms
from PIL import Image
import dehaze22  as net
import torchvision.utils as vutils
# import matplotlib.pyplot as plt
import os
# import cv2

def col_and_row(image_name):
        # 'save_1_0.jpg' is the example format
    col_num = int(image_name.split("_")[1])
    row_num = image_name.split("_")[2]
    row_num = int(row_num.split(".")[0])
    return col_num, row_num


#%%
if not os.path.exists("./res"):
    raise Exception("Tiled images cannot found!") 
        

im_arr = sorted(os.listdir("./res"))

    
# First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
# data_transform = transforms.Compose([transforms.Resize((1536, 2048)), transforms.ToTensor()])
data_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
redata_transform = transforms.Compose([transforms.Resize((450, 800))])
# data_transform = transforms.Compose([transforms.ToTensor()])

print("Model is loading.")

netG = net.dehaze(3, 3, 64)

# netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cpu')))
netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cuda')))
netG.cuda()


print("Inference has been started.")

for images in im_arr:
    

    image = Image.open('./res/' + images)
    H, W = image.height, image.width
    image = data_transform(image).unsqueeze(0).cuda() # For GPU
    # image = data_transform(image).unsqueeze(0) # For CPU
    
    image = (image - image.min())/ (image.max() - image.min())
    
    with torch.no_grad():
        torch.cuda.synchronize() 
        start = time.time()
        x_hat, _, _, _= netG(image)
        # output1 = x_hat[0].cpu().permute(1,2,0).detach().numpy()
        # output = np.interp(output1, (output1.min(), output1.max()),(0,255))   
        # output = cv2.resize(output.astype(np.uint8), (W,H))
        # cv2.imwrite('./res_res/' + images, output)
        torch.cuda.synchronize() 
        final_time = time.time() - start
        out = redata_transform(x_hat[0])
        vutils.save_image(out,'./res_res/' + images, normalize=True, scale_each=False)

        print("Inference time is {} seconds".format(str(final_time)))
        print(images + " has been dehazed.")
# cv2.imwrite("final_nonorm_3.jpg", final_im)
    
    
    
    
    
    
    
    
