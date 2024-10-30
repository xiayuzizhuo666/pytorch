import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
from model import *

from torch import nn
from torch.utils.data import DataLoader
train_data = torchvision.datasets.CIFAR10(root="datasat", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="datasat", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

#长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#dataloader加载
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#神经网络搭建 10
#model xiayu

xiayu = Xiayu()


#loss 损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器 optimizer
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(xiayu.parameters(), lr=learning_rate)

#设置训练网络参数
total_train_step = 0 #记录训练次数
total_test_step = 0 #记录测试次数
epoch = 20 #训练轮数

#tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("!----第{}轮训练开始----!".format(i+1))
    for data in train_dataloader:
        imgs,targets = data
        outputs = xiayu(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()#调用优化器
        loss.backward()#反向传播
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{},Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
#测试开始
    xiayu.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = xiayu(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    torch.save(xiayu, "xiayu_{}.pth".format(i))
    print("模型已保存")

writer.close()