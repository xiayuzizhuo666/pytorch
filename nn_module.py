from torch import nn
import torch

class Xiayu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


xiayu = Xiayu()
x = torch.tensor(1.0)
output = xiayu(x)
print(output)
