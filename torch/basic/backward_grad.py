import torch

x = torch.tensor(-3., requires_grad=True)
y = torch.tensor(2., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

q = x+y

f = q*z

f.backward()

print(z.grad)
