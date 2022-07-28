import numpy as np
import cv2
from mss import mss
import torch
import dehaze22  as net


print("Model is loading...")
netG = net.dehaze(3, 3, 64)

netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cuda'))) # For GPU
netG.cuda() # For GPU
# netG.load_state_dict(torch.load('netG.pth', map_location=torch.device('cpu'))) # For CPU

print("Model has been loaded.")
print("Press 'q' to quit.")

bounding_box = {'top': 100, 'left': 100, 'width': 512, 'height': 512}
sct = mss()

while True:
    sct_img = sct.grab(bounding_box) #take the screenshot
    
    np_frame = np.array(sct_img)[:,:,0:3].T   # convert it to numpy array
    np_frame = (np_frame - np_frame.min())/(np_frame.max()-np_frame.min()) # normalize array
    frame = torch.tensor(np_frame, dtype=torch.float32) # convert it to torch tensor
    
    frame = frame.unsqueeze(0).cuda() # For GPU
    # frame = frame.unsqueeze(0) # For CPU
    
    with torch.no_grad():
        out_frame , _, _, _ = netG(frame)
        show = out_frame[0].cpu().permute(2,1,0)
        
        cv2.imshow('screen', np.array(show))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
