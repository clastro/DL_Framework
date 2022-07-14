import torch
import torch.nn as nn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(10,20)
    self.fc2 = nn.Linear(20,20)
    self.output = nn.Linear(20,4)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.output(x)
    return x
