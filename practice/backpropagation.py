#곱셈 노드 순전파, 역전파

class MulLayer:
  def __ init__(self):
    self.x = None
    self.y = None
    
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out
    
  def backward(self,dout):
    dx = dout * self.y
    dy = dout * self.x
    return dx,dy
 
#덧셈 노드 순전파, 역전파

class AddLayer:
  def __ init__(self):
    pass
    
  def forward(self, x, y):
    out = x + y
    return out
    
  def backward(self,dout):
    dx, dy = dout * 1
    return dx,dy
    
    
