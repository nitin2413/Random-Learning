import torch

a = torch.eye(3)
print(a)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([3.0, 2.0, 1.0])

dot = torch.dot(x,y)
print(dot)

m = torch.arange(0,12).reshape(3,4)
print(m)

first_row = m[0]
element = m[0,3]
sub_matrix = m[:,1:3]
m_t = m.T
print(m_t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.tensor([1.0,2.0], device=device)
print(a)
x = torch.tensor([2.0], requires_grad=True)
y = x**2+3*x+1
y.backward()
print(x.grad)
a = torch.tensor([1.0,2.0,3.0], requires_grad= True)
b = a * 2
c = b.mean()
c.backward()
print(a.grad)