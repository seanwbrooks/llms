import torch
import torch.nn as nn
import tiktoken
import numpy as np
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt"
)

model_configs = {
    "gpt2-small (124M)": { "emb_dim": 768, "n_heads": 12, "n_layers": 12 },
    "gpt2-medium (355M)": { "emb_dim": 1024, "n_heads": 16, "n_layers": 24 },
    "gpt2-large (774M)": { "emb_dim": 1280, "n_heads": 20, "n_layers": 36 },
    "gpt2-xl (1558M)": { "emb_dim": 1600, "n_heads": 25, "n_layers": 48 }
}

GPT_CONFIG_124M = {
	"vocab_size": 50257, # Vocabulary size (i.e. The Verdict)
	"context_length": 1024, # Context size
	"emb_dim": 768, # Embedding dimension
	"n_heads": 12, # Number of attention heads
	"n_layers": 12, # Number of layers
	"drop_rate": 0.1, # Dropout rate (for preventing overtraining weights)
	"qkv_bias": False
}

class MultiHeadAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
		super().__init__()
		assert (d_out % num_heads == 0), \
			"d_out must be divisible by num_heads"
		self.d_out = d_out
		self.num_heads = num_heads
		self.head_dim = d_out // num_heads
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.out_proj = nn.Linear(d_out, d_out)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

	def forward(self, x):
		b, num_tokens, d_in = x.shape
		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)

		keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # TODO: Debug RuntimeError: shape '[2, 6, 2, 2]' is invalid for input of size 24
		values = values.view(b, num_tokens, self.num_heads, self.head_dim)
		queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

		# Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
		keys = keys.transpose(1, 2)
		queries = queries.transpose(1, 2)
		values = values.transpose(1, 2)

		attn_scores = queries @ keys.transpose(2, 3) # Computes dot product for each head
		mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # Masks truncated to the number of tokens

		attn_scores.masked_fill_(mask_bool, -torch.inf) # Uses the mask to fill attention scores
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)

		context_vec = (attn_weights @ values).transpose(1, 2) # Tensor shape: (b, num_tokens, n_heads, head_dim)
		context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
		context_vec = self.out_proj(context_vec)
		return context_vec

class GELU(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return 0.5 * x * (1 + torch.tanh(
			torch.sqrt(torch.tensor(2.0 / torch.pi)) *
			(x + 0.044715 * torch.pow(x, 3))
		))

class FeedForward(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
			GELU(),
			nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
		)
	def forward(self, x):
		return self.layers(x)

class LayerNorm(nn.Module):
	def __init__(self, emb_dim):
		super().__init__()
		self.eps = 1e-5
		self.scale = nn.Parameter(torch.ones(emb_dim))
		self.shift = nn.Parameter(torch.zeros(emb_dim))

	def forward(self, x):
		mean = x.mean(dim=-1, keepdim=True)
		var = x.var(dim=-1, keepdim=True, unbiased=False)
		norm_x = (x - mean) / torch.sqrt(var + self.eps)
		return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.att = MultiHeadAttention(
			d_in=cfg["emb_dim"],
			d_out=cfg["emb_dim"],
			context_length = cfg["context_length"],
			num_heads = cfg["n_heads"],
			dropout = cfg["drop_rate"],
			qkv_bias = cfg["qkv_bias"],
		)
		self.ff = FeedForward(cfg)
		self.norm1 = LayerNorm(cfg["emb_dim"])
		self.norm2 = LayerNorm(cfg["emb_dim"])
		self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
	def forward(self, x):
		shortcut = x
		x = self.norm1(x)
		x = self.att(x)
		x = self.drop_shortcut(x)
		x = x + shortcut
		shortcut = x
		x = self.norm2(x)
		x = self.ff(x)
		x = self.drop_shortcut(x)
		x = x + shortcut
		return x

class GPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
		self.drop_emb = nn.Dropout(cfg["drop_rate"])

		self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
		
		self.final_norm = LayerNorm(cfg["emb_dim"])
		self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
	def forward(self, in_idx):
		batch_size, seq_len = in_idx.shape
		tok_embeds = self.tok_emb(in_idx)
		pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
		x = tok_embeds + pos_embeds
		x = self.trf_blocks(x)
		x = self.final_norm(x)
		logits = self.out_head(x)
		return logits

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG["context_length"] = 1024
NEW_CONFIG["qkv_bias"] = True
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: Left {left.shape}, Right {right.shape}")
	
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
	
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
	
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
	
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
    
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
	
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
	for _ in range(max_new_tokens):
		idx_cond = idx[:, -context_size:]
		with torch.no_grad():
			logits = model(idx_cond)
		logits = logits[:, -1, :]
		if top_k is not None:
			top_logits, _ = torch.topk(logits, top_k)
			min_val = top_logits[:, -1]
			logits = torch.where(
				logits < min_val,
				torch.tensor(float('-inf')).to(logits.device),
				logits
			)
		if temperature > 0.0:
			logits = logits / temperature
			probs = torch.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
		else:
			idx_next == torch.argmax(logits, dim=-1, keepdim=True)
		if idx_next == eos_id:
			break
		idx = torch.cat((idx, idx_next), dim=1)
	return idx

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
	encoded_tensor = torch.tensor(encoded).unsqueeze(0)
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0)
	return tokenizer.decode(flat.tolist())

tokenizer = tiktoken.get_encoding("gpt2")
	
load_weights_into_gpt(gpt, params)
device = torch.device("mps" if torch.mps.is_available() else "cpu")
gpt.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Don Gately was a ", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))