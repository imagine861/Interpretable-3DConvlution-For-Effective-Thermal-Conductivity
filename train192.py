import argparse
import time
from dataloader import tifDataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
#from net import C3D
from net_cam192 import C3D

#加入命令行参数，主要是包括数据集的位置信息
parser = argparse.ArgumentParser(description='Training For DL Methods Predict Thermal Conductivity Of Materials')
parser.add_argument('-d', '--data', default='./data',
                    help='path to dataset')

# 运行代码
if __name__ == '__main__':
    args = parser.parse_args()

    #设置cuda的运行环境
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #插入时间帧记录代码运行开始时间
    start_time = time.time()

    #通过Dataset和Loader两个工具加载数据集
    trainset = tifDataset('train', args)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    valset = tifDataset('val', args)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=0)

    #写入训练日志
    with open('log/train_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss"])

    #指定GPU运行设备的ID
    device_ids = [0]

    # 模型搭建,损失函数以及优化器及学习率更新策略选择
    model = nn.DataParallel(C3D(), device_ids=device_ids)
    #模型加载并导入到cuda#模型加载并导入到cuda
    criterion = nn.L1Loss()#选择Mse的损失函数
    #优化器选择SGD，其中学习率初始值为0.1，L2正则项值为0.001，动量选择0.9
    optimizer = optim.SGD(model.parameters(), lr=10e-4, weight_decay=10e-5, momentum=0.9)
    #学习率的更新策略选择更加平滑的指数衰减，更新速率为0。9
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    #初始化所有的变量
    i = 0
    val_loss = 0
    max_val_loss = 0

    #定义模型评估验证集的方法，返回值为验证集上的误差
    def test(model, testloader, criterion):
        # test model on testloader
        # return val_loss

        model.eval()

        loss,count = 0,0.

        with torch.no_grad():
            for (images, labels) in testloader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                count += 1

        return loss/count

    # 开始训练
    train_loss, counter = 0, 0.

    #选择合适的迭代次数进行训练
    for epoch in range(100):

        epoch_start_time = time.time()


        for data in trainloader:

            #模型训练
            model.train()

            #将数据导入cuda并喂给模型
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            #得到模型的输出值
            outputs = model(inputs)

            #前向传播进行计算Loss值，并通过反向传播更新模型权重
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            counter += 1

        # 记录本次迭代的训练和验证损失
        train_loss /= counter
        val_loss= test(model,valloader,criterion)
        print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f' % (i, epoch, train_loss, val_loss))

        with open('log/train_log_16_16_32.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, train_loss, val_loss])

        #设置监控验证集的损失来进行保存最佳的模型权重
        if val_loss> max_val_loss:
            torch.save(model.state_dict(), 'weight/weights_1.pkl')
            max_val_acc = val_loss
        if epoch%5 == 0:
                    torch.save(model.state_dict(), 'weight/weights_'+str(epoch)+'.pkl')

        #重置训练过程的变量为了进行下一步的迭代
        train_loss, counter = 0, 0
        i += 1
        print("epoch time %.4f min" % ((time.time() - epoch_start_time) / 60))
        ExpLR.step()

    #记录训练完成并打印一个epoch所需时间
    train_time = time.time()
    print("train time %.2f h" % ((train_time - start_time) / 3600))


