class XOR_model:
    
    def __init__(self, lr=0.01, num_epochs = 1000):
        
        self.W = np.random.rand(4).reshape(2,2)
        self.b = np.random.rand(2).reshape(2,1)
        self.W2 = np.random.rand(2).reshape(2,1)
        self.b2 = np.random.rand(1,1)
        self.X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
        self.y = np.array([0,1,1,0]).T
        self.loss_score = []
        self.num_epochs = num_epochs
        self.lr = lr
        
    def activation(self,x): #sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def forward(self,x):
        
        self.h = np.dot(self.W.T, x) + self.b
        self.h_out = self.activation(self.h)

        self.output = np.dot(self.W2.T, self.h_out) + self.b2
        self.y_hat = self.activation(self.output)
        
        return self.y_hat
    
    def backward(self):
        
        error = (self.y - self.y_hat)        
        loss = 0.5 * np.square(error)
        self.loss_score.append(np.sum(loss))
        
        grad2 = np.dot(self.h_out,(error).T)
        
        dH = np.dot(self.W2, error)
        dF = dH * self.h_out * (1 - self.h_out)
        grad = np.dot(self.X, dF.T)        
        
        self.W += self.lr * grad
        self.W2 += self.lr * grad2  
        
        self.b += np.sum(self.lr * error * self.y_hat * (1 - self.y_hat)) #틀렸는데 잘 되고 있음  
        self.b2 += np.sum(self.lr * error * self.y_hat * (1 - self.y_hat)) #틀렸는데 잘 되고 있음
    
    def train(self):
        for i in range(self.num_epochs):
            if(i==1):
                print('초기값 : ',self.y_hat)
            self.forward(self.X)
            self.backward()
            
        print('결과 : ',self.y_hat)
