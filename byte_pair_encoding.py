# Required `pip install tiktoken`
from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"))

with open("./text-files/text-for-data-prep.txt", "r", encoding="UTF-8") as f:
	text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# A dataset for batched inputs and targets
from torch.utils.data import Dataset, DataLoader
import torch

class GPTDatasetV1(Dataset):
	def __init__(self, txt, tokenizer, max_length, stride):
		self.input_ids = []
		self.target_ids = []

		token_ids = tokenizer.encode(txt)

		for i in range(0, len(token_ids) - max_length, stride):
			input_chunk = token_ids[i:i + max_length]
			target_chunk = token_ids[i + 1: i + max_length + 1]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.target_ids[idx]

# A data loader to generate batches with input-with pairs
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
	tokenizer = tiktoken.get_encoding("gpt2")
	dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

	return dataloader

# call dataloader with sample text file
dataloader = create_dataloader_v1(text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(text, batch_size=8, max_length=16, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

