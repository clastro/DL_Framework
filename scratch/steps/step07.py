class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
        
class Function(self, input):
    x = input.data
    y = self.forward(x)
    output = Variable(y)
    output.set_creator(self)
    self.input = input
    self.output = output
    return output
