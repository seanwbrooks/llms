import torch
import time

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

torch.manual_seed(123)
X = torch.rand((1,50))
out = model(X)
print(out)

with torch.no_grad():
	out = model(X)
print(out)

with torch.no_grad():
	out = torch.softmax(model(X), dim=1)
print(out)

## Setting up efficient data loaders
# Creating a small toy dataset
X_train = torch.tensor([[-1.2, 3.1],[-0.9, 2.9],[-0.5, 2.6],[2.3, -1.1],[2.7, -1.5]])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([[-0.8, 2.8],[2.6, -1.6]])
y_test = torch.tensor([0, 1])

# Defining a custom Dataset class
from torch.utils.data import Dataset

class ToyDataset(Dataset):
	def __init__(self, X, y):
		self.features = X
		self.labels = y
	def __getitem__(self, index):
		one_x = self.features[index]
		one_y = self.labels[index]
		return one_x, one_y
	def __len__(self):
		return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print('Length of train dataset ', len(train_ds))

# Instantiating data loaders
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)

test_loader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)

for idx, (x, y) in enumerate(train_loader):
	print(f"Batch {idx+1}: ", x, y)

# A training loader that drops the last batch
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

for idx, (x, y) in enumerate(train_loader):
	print(f"Batch {idx+1}: ", x, y)

# Neural Network training in PyTorch
torch.manual_seed(123)

model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
	model.train()

	for batch_idx, (features, labels) in enumerate(train_loader):
		logits = model(features)
		
		loss = F.cross_entropy(logits, labels)
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		### Logging
		print(
			f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
			f" | Batch: {batch_idx:03d}/{len(train_loader):03d}"
			f" | Train Loss: {loss:.2f}"
		)
	model.eval()

# A function to compute the pediction accuracy
def compute_accuracy(model, dataloader):
	model = model.eval()
	correct = 0.0
	total_examples = 0

	for idx, (features, labels) in enumerate(dataloader):
		with torch.no_grad():
			logits = model(features)

		predictions = torch.argmax(logits, dim=1)
		compare = labels == predictions
		correct += torch.sum(compare)
		total_examples += len(compare)

	return (correct / total_examples).item()

print('Compute accuracy ', compute_accuracy(model, train_loader))

torch.save(model.state_dict(), "model.pth")

# Running training loop on GPU
print(torch.cuda.is_available())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# A training loop on a GPU
start_time = time.perf_counter()
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

device = torch.device("mps")
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
	
	model.train()
	for batch_idx, (features, labels) in enumerate(train_loader):
		features, labels = features.to(device), labels.to(device)
		logits = model(features)
		loss = F.cross_entropy(logits, labels) # Loss function

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		### LOGGING
		print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
		      f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
		      f" | Train/Val Loss: {loss:.2f}")

		model.eval()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
# Evaluate performance
print(f"MPS elapsed time: {elapsed_time}")

# A training loop on the CPU
start_time = time.perf_counter()
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

device = torch.device("cpu")
print(device)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
	model.train()
	for batch_idx, (features, labels) in enumerate(train_loader):
		features, labels = features.to(device), labels.to(device)
		logits = model(features)
		loss = F.cross_entropy(logits, labels) # Loss function

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		### LOGGING
		print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
		      f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
		      f" | Train/Val Loss: {loss:.2f}")

		model.eval()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
# Evaluate performance
print(f"CPU elapsed time: {elapsed_time}")

