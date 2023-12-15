from net_cam192 import C3D
import glob
import torch
import os
import pandas as pd
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import tifffile
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
model = nn.DataParallel(C3D(), device_ids=device_ids)
model.cuda(device=device_ids[0])
model.load_state_dict(torch.load('./weight/weights_best_0407.pkl'))



img = tifffile.imread('./data/test/0.tif')
# torch.from_numpy(img).unsqueeze(0).float()
img = torch.FloatTensor(img)
img = torch.unsqueeze(img, 0)
img = torch.unsqueeze(img,0).cuda()
with torch.no_grad():
    pred = model(img).cpu().detach().numpy()[0]
    print(pred)





