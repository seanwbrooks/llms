import torch
import importlib
from generative_pretrained_transformer_model import GPTModel, generate_text_simple
import tiktoken
from byte_pair_encoding import create_dataloader_v1

file_path = "./text_files/infinite-jest.txt"
with open(file_path, "r", encoding="utf-8") as f:
	text_data = f.read()

GPT_CONFIG_124M = {
	"vocab_size": 50257,
	"context_length": 256,
	"emb_dim": 768,
	"n_heads": 12,
	"n_layers": 12,
	"drop_rate": 0.1,
	"qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	encoded_tensor = torch.tensor(encoded).unsqueeze(0)
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0)
	return tokenizer.decode(flat.tolist())

start_context = "Every time man"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
	model=model,
	idx=text_to_token_ids(start_context, tokenizer),
	max_new_tokens=10,
	context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Here we have generated incoherent text, because of no training.

# calculating the text generation loss
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

with torch.no_grad():
	logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1: ", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2: ", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("Log probas: ", log_probas)

avg_log_probas = torch.mean(log_probas)
print("Average log probs: ", avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print("Negative average log probability: ", neg_avg_log_probas)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits: ", logits_flat.shape)
print("Flattened targets: ", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters: ", total_characters)
print("Tokens: ", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
	train_data,
	batch_size=2,
	max_length=GPT_CONFIG_124M["context_length"],
	stride=GPT_CONFIG_124M["context_length"],
	drop_last=True,
	shuffle=True,
	num_workers=0
)
val_loader = create_dataloader_v1(
	val_data,
	batch_size=2,
	max_length=GPT_CONFIG_124M["context_length"],
	stride=GPT_CONFIG_124M["context_length"],
	drop_last=True,
	shuffle=True,
	num_workers=0
)
print("Train loader: ")
for x, y in train_loader:
	print(x.shape, y.shape)

print("\nValidation loader: ")
for x, y in val_loader:
	print(x.shape, y.shape)
