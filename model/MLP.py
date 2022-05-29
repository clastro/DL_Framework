import torch

print(torch.__version__)

class MLP(Module):
  # init
  def __init__(self,n_inputs):
    super(MLP, self).__init__()
    self.layer = Linear(n_inputs, 1) #input은 n_inputs으로 개수를 설정하고 output은 1개인 Linear layer
    self.activation = sigmoid() #activation 변수에는 sigmoid를 사용
  # forward propagation
  def forward(self, X):
    X = self.layer(X)
    X = self.activation(X)
    return X
    
  
