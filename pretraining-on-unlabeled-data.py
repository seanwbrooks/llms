import torch
import importlib
gptmodel = importlib.import_module("generative-pretrained-transformer-model") 
import tiktoken

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
model = gptmodel.GPTModel(GPT_CONFIG_124M)
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

token_ids = gptmodel.generate_text_simple(
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
