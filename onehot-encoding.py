import torch

def to_onehot(y, num_classes):
	y_onehot = torch.zeros(y.size(0), num_classes)
	y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
	return y_onehot

y = torch.tensor([0, 1, 2, 2])

y_enc = to_onehot(y, 3)

print('one-hot encoding:\n', y_enc)
