import os.path

from slicegan import model, networks, util
def Generator(save_path,i,j):
    Project_name = 'fib-png-3-twophase-30epoch-PF30-69_7-medium-opening-no_seperate_object-0_pore-1_resin'
    Project_dir = './'
    Training = False
    Project_path = util.mkdr(Project_name, Project_dir, Training)
    image_type = 'twophase'
    img_size, img_channels, scale_factor = 64, 2,  1
    z_channels = 16
    lays = 6
    dk, gk = [4]*lays, [4]*lays
    ds, gs = [2]*lays, [2]*lays
    # no. filters
    df, gf = [img_channels,64,128,256,512,1], [z_channels,512,256,128,64,img_channels]
    # paddings
    dp, gp = [1,1,1,1,0],[2,2,2,2,3]

    ## Create Networks
    netD, netG = networks.slicegan_nets(Project_path, Training, image_type, dk, ds, df, dp, gk, gs, gf, gp)

    #img,image_trans,raw, netG = util.test_img(Project_path, image_type,image_type_trans,netG(), z_channels, lf=25, periodic=[0, 1, 1])
    util.test_img(save_path, i,j, image_type, netG(), z_channels, lf=11, periodic=[0, 0, 0])

if __name__  == '__main__':
    for j in range(5):
        for i in range(1,10):
            path = './data/heatmap11/'
            if os.path.exists(path):
                pass
            else:
                os.makedirs(path)
            Generator(path,j,i)

