from net_cam192 import C3D
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile

device_ids = [0]
model = nn.DataParallel(C3D(), device_ids=device_ids)
model.cuda(device=device_ids[0])
model.load_state_dict(torch.load('./weight/weights_best.pkl'))
img_path = './data/heatmap/0.tif'


img = tifffile.imread(img_path)
img = torch.FloatTensor(img)
img = torch.unsqueeze(img, 0)
img = torch.unsqueeze(img, 0).cuda()

params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
# pd.DataFrame(weight_softmax).to_excel('weight.xlsx')
features_blobs = []
def hook_feature(module, input, output):
     features_blobs.append(output.data.cpu().numpy())

model.module.group5.register_forward_hook(hook_feature)

output = model(img)
print(output)
weight_softmax = weight_softmax.reshape((64,1))
print(weight_softmax)
#
for i in range(len(features_blobs[0][0])):
    tifffile.imwrite('./cam/' + str(i) + '.tif', features_blobs[0][0][i])
#
featuremap = glob.glob('./cam/*.tif')
#
feature = np.zeros((64,192,192,192))
#
for i in range(len(featuremap)):
    feature[i] = tifffile.imread(featuremap[i])
#
cam = weight_softmax.T.dot(feature.reshape(64,192*192*192))
cam = cam.reshape(192,192,192)
cam = cam - np.min(cam)
cam_img = cam / np.max(cam)
cam_img = np.uint8(255 * cam_img)
#
tifffile.imwrite('./heatmap/heatmap.tif',cam_img)


for i in range(192):
    heatmap = cv2.applyColorMap(cam_img[i], cv2.COLORMAP_HOT)
    cv2.imwrite('./heatmap/heatmap%s.jpg'%(i+1),heatmap)



