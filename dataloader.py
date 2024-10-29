import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备测试集
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

#测试集第一章
img,target = test_data[0]
print(img.shape)
print(target)

writer =SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(target)
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step = step + 1

writer.close()





