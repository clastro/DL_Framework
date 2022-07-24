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

import torch.nn.functional as F

# Define another class Net
class Net(nn.Module):
    def __init__(self):    
    	# Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):   
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
