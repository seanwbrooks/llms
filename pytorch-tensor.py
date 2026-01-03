import torch

# Common PyTorch tensor operations
tensor2d = torch.tensor([[1,2,3],[4,5,6]])

print('tensor2d ', tensor2d)

print('tensor2d shape ', tensor2d.shape)

torch.Size([2,3])

print('tensor2d reshape to 3x2 ', tensor2d.reshape(3,2))

print('tensor2d view to 3x2 ', tensor2d.view(3,2))

print('tensor2d tranpose ', tensor2d.T)

print('tensor2d multiple 2 matrices ', tensor2d.matmul(tensor2d.T))

print('multiple in different way ', tensor2d @ tensor2d.T)

# Computational graphs

import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

# Computing gradients via autograd
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print('gradient of loss of weight ', grad_L_w1)
print('gradient of loss of bias ', grad_L_b)

loss.backward()
print(w1.grad)
print(b.grad)

# A multilayer perceptron with two hidden layers
class NeuralNetwork(torch.nn.Module):
	def __init__(self, num_inputs, num_outputs):
		super().__init__()

		self.layers = torch.nn.Sequential (
			# 1st hidden layer
			torch.nn.Linear(num_inputs, 30),
			torch.nn.ReLU(),
			# 2nd hidden layer
			torch.nn.Linear(30, 20),
			torch.nn.ReLU(),
			# output layer
			torch.nn.Linear(20, num_outputs),
		)
	def forward(self, x):
		logits = self.layers(x)
		return logits

model = NeuralNetwork(50, 3)

print('Model ', model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of trainable model parameters: ', num_params)

print('Weight parameter ', model.layers[0].weight)
print('Dimensions of weight using .shape ', model.layers[0].weight.shape)

