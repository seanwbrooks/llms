import torch
inputs = torch.tensor([
	[0.43, 0.15, 0.89],
	[0.55, 0.87, 0.66],
	[0.57, 0.85, 0.64],
	[0.22, 0.58, 0.33],
	[0.77, 0.25, 0.10],
	[0.05, 0.80, 0.55]
])

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
	attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())

def softmax_naive(x):
	return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights: ", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
	context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

# Implementing a compact self-attention class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
	def __init__(self, d_in, d_out):
		super().__init__()
		self.W_query = nn.Parameter(torch.rand(d_in, d_out))
		self.W_key = nn.Parameter(torch.rand(d_in, d_out))
		self.W_value = nn.Parameter(torch.rand(d_in, d_out))
	def forward(self, x):
		keys = x @ self.W_key
		queries = x @ self.W_query
		values = x @ self.W_value
		attn_scores = queries @ keys.T # omega
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		context_vec = attn_weights @ values
		return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# A self-attention class with Linear Layers
class SelfAttention_v2(nn.Module):
	def __init__(self, d_in, d_out, qkv_bias=False):
		super().__init__()
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

	def forward(self, x):
		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)
		attn_scores = queries @ keys.T
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		context_vec = attn_weights @ values
		return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

batch = torch.stack((inputs, inputs), dim=0)
print(batch)

torch.Size([2, 6, 3])

# A compact casual attention class
class CasualAttention(nn.Module):
	def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
		super().__init__()
		self.d_out = d_out
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

	def forward(self, x):
		b, num_tokens, d_in = x.shape
		keys = self.W_key(x)
		queries = self.W_query(x)
		values = self.W_value(x)
		
		attn_scores = queries @ keys.transpose(1, 2)
		attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)

		context_vec = attn_weights @ values
		return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape: ", context_vecs.shape)
