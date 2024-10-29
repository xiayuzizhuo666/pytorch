import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype=torch.float32)#不加这也不会报错，2024已经兼容
#
# input=torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class Xiayu(nn.Module):
    def __init__(self):
        super(Xiayu, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        return self.maxpool1(input)
        return output

xiayu = Xiayu()
# output = xiayu(input)
# print(output)

writer = SummaryWriter("logs_maxpool")
step = 0

for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    outputs = xiayu(imgs)
    writer.add_images("output",outputs,step)
    step = step + 1

writer.close()