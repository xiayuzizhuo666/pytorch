import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

])
train_set = torchvision.datasets.CIFAR10(root="dataset", train=True, transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset", train=False, transform=dataset_transform,download=True)

# print(test_data[0])
# print(train_data.classes)
#
# img,target = test_data[0]
# print(img)
# print(target)
# print(test_data.classes[target])
# img.show()
#
# print(train_data[0])

writer = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_data",img,i)
writer.close()