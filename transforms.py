from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from writer_Tensorboard import writer

img_path = "data/train/ants_image/0013035.jpg"
img =Image.open(img_path)

SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("tensor", tensor_img)
# print(tensor_img)

writer.close()