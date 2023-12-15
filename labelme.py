import os
import csv
from shutil import move
import numpy as np
import pumapy as puma


img_path = './data/raw_train/'
img = os.listdir(img_path)

trainset_img =  './data/train/'
if not os.path.exists(trainset_img):
    os.makedirs(trainset_img)

for idex,img_name in enumerate (img):
    move(img_path+str(img_name),trainset_img+str(idex+16308+360*1)+'.tif')
    materials = puma.import_3Dtiff(trainset_img+str(idex+16308+360*1)+'.tif')

    cond_map = puma.IsotropicConductivityMap()
    cond_map.add_material((0, 89), 0.0257)
    cond_map.add_material((90, 255), 0.2)

    k_eff_z, T_z, q_z = puma.compute_thermal_conductivity(materials, cond_map, 'x', 'p', tolerance=1e-5, solver_type='cg')
    with open(trainset_img + 'label.txt', 'a',newline='') as f:
        f.writelines([str(idex+16308+360*1)+'.tif ',str(k_eff_z[0]) + '\n'])
        f.close()
