class Function:
    def __call__(self,input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()

class Square(Function): #Function을 상속하고 forward 함수 정의
    def forward(self,x):
        return x ** 2
