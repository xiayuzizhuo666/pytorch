from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer: SummaryWriter = SummaryWriter("logs")
image_path = "data/train/ants_image/6240338_93729615ec.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("test", img_array,3,dataformats='HWC')
# y=x

for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()