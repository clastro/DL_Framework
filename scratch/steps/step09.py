def as_array(x): #scalar to np.array
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은 지원하지 않습니다.".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) #역전파 최초값은 1로 설정 
            
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f,output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)
        
class Function(self, input):
    x = input.data
    y = self.forward(x)
    output = Variable(as_array(y))
    output.set_creator(self)
    self.input = input
    self.output = output
    return output

class Square(Function):
    def forward(self,x):
        return x ** 2
    def backward(self,gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self,x):
        return np.exp(x)
    
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx 