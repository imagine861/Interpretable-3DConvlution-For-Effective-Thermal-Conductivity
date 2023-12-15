import tifffile
import torch

# train_loader = torch.utils.data.DataLoader()

class tifDataset(torch.utils.data.Dataset):
    def __init__(self,mode,args):
        self.mode =mode
        self.data_path = args.data
        self.img_path,self.label =self.data_init(self.data_path)

    def data_init(self,data_path):
        label_path = data_path + '/label.txt'
        with open(label_path) as f:
            lines = f.readlines()

        img_path = []
        label = []

        for line in lines:
            line_sp = line.split()
            img_name = line_sp[0]
            img_label = float(line_sp[1])
            img_path.append(data_path + '/train/' + str(img_name))

            label.append(img_label)

        num_train = int(0.8*len(img_path))

        if self.mode == 'train':
            img_path = img_path[:num_train]
            label= label[:num_train]
        elif self.mode == 'val':
            img_path = img_path[num_train:]
            label = label[num_train:]

        return img_path,label

    def __len__(self):

        return len(self.img_path)

    def __getitem__(self, index):


        img_path = self.img_path[index]
        img = tifffile.imread(img_path)
        # torch.from_numpy(img).unsqueeze(0).float()
        img = torch.FloatTensor(img)
        img = torch.unsqueeze(img, 0)

        label = self.label[index]
        label = torch.tensor(label)

        return img, label

