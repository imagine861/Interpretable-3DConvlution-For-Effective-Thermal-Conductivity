import os.path
import glob
from net_cam1 import C3D
import torch
import pandas as pd
import numpy as np
import cv2
import tifffile

def ResultandHeatmap(img_path,model):
    img = tifffile.imread(img_path)
    img = torch.FloatTensor(img)
    img = torch.unsqueeze(img, 0)
    img = torch.unsqueeze(img, 0).cuda()

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    pd.DataFrame(weight_softmax).to_excel('weight.xlsx')


    features_blobs = []
    def hook_feature(module, input, output):
          features_blobs.append(output.data.cpu().numpy())

    model.group3.register_forward_hook(hook_feature)

    ouput = model(img)

    feature = features_blobs[0]

    cam = weight_softmax.T.dot(feature.reshape(32, 320 * 320 * 320))
    cam = cam.reshape(320, 320, 320)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    if not os.path.exists('./cam_img/'):
        os.makedirs('./cam_img/')
    tifffile.imwrite('./cam_img/cam'+str(img_path[-8:-4])+'.tif',cam_img)
    heatmap = np.zeros((320, 320, 320, 3))
    for i in range(320):
        heatmap[i] = cv2.applyColorMap(cam_img[i], cv2.COLORMAP_JET)

    tifffile.imwrite('./cam_img/heatmap' + str(img_path[-8:-4]) + '.tif', cam_img)

    for i in range(320):
        if not os.path.exists('./cam_img/heatmap' + str(img_path[-8:-4]) + '/'):
            os.makedirs('./cam_img/heatmap' + str(img_path[-8:-4]) + '/')
        cv2.imwrite('./cam_img/heatmap' + str(img_path[-8:-4]) + '/'+ str(i)+'.jpg',heatmap[i])


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    model = C3D().eval().to(device)
    model.load_state_dict(torch.load('./weight/weights_1.pkl', map_location=device))
    img_path = glob.glob('./data/test/')

    for img in img_path:
        ResultandHeatmap(img_path,model)






