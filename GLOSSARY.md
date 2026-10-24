# Building an LLM

The vocabulary for building a GPT-style language model step by step: turning text into numbers, the model's architecture, pretraining, and generating text.

## Text → numbers

**Token**:
One unit of text as the tokenizer sees it. Can be a whole word, part of a word, a punctuation mark or a special token.
_Avoid_: word

**Token ID**:
The integer that stands for a token in a vocabulary.
_Avoid_: index, integer

**Tokenizer**:
A two-way mapping between text and token IDs (encode and decode).

**Byte pair encoding (BPE)**:
A tokenizer scheme that builds subword tokens by repeatedly merging the most frequent pairs. It can encode any text without an unknown-token fallback. GPT-2 uses it.

**Vocabulary**:
The fixed set of tokens a tokenizer knows, each with a token ID. The vocabulary belongs to the tokenizer, not to any one text. GPT-2's vocabulary has 50,257 tokens.
_Avoid_: vocab size of a text (that is either the token count or the number of unique tokens)

**Token count**:
How many tokens a particular text encodes to.
_Avoid_: vocabulary size

**Special token**:
A reserved token that marks structure, not content, such as `<|endoftext|>` between documents or `<|unk|>` for unknown words.

## Training data

**Input–target pair**:
A sequence of token IDs and the same sequence shifted one position to the right. At every position, the target is the next token.

**Training window**:
The number of tokens in each input of an input–target pair. It can be at most the context length.
_Avoid_: max_length (the code's parameter name), context size

**Stride**:
How far the training window moves forward between consecutive pairs. If stride equals the window, the pairs don't overlap.

**Batch**:
Several input–target pairs stacked together and processed in one step.

**Training set / Validation set**:
The disjoint parts of the corpus used to update weights and to measure generalisation.
_Avoid_: test set (no held-out test split exists)

## Model architecture

**Context length**:
The largest number of tokens the model can attend to at once. It is fixed by the number of positional embeddings (1,024 for GPT-2).
_Avoid_: context size, sequence length

**Embedding dimension**:
The width of the vector that represents each token as it passes through the model.
_Avoid_: hidden size, d_model

**Token embedding**:
A learned vector looked up for each token ID.

**Positional embedding**:
A learned vector for each position in the context, added to the token embedding.

**Causal attention**:
Self-attention in which each token can attend only to itself and earlier tokens.
_Avoid_: casual attention, masked attention

**Attention head**:
One independent set of query, key and value projections inside multi-head attention.

**Multi-head attention**:
Several attention heads run in parallel and then concatenated into one output.

**Transformer block**:
The repeated unit of the model: layer norm, causal multi-head attention, a feed-forward network and shortcut connections. GPT-2 sizes are described by how many blocks they stack.
_Avoid_: layer (except in the config key `n_layers`, which counts blocks)

**Shortcut connection**:
Adding a sub-layer's input back to its output, so that gradients can flow through deep stacks.
_Avoid_: skip connection, residual (same idea; this repo uses the book's term)

**Logits**:
The model's raw, unnormalised score for every vocabulary token at each position.

**Output head**:
The final projection from the embedding dimension to the vocabulary size, which produces logits.

## Training & weights

**Pretraining**:
Training a model from random weights on unlabeled text to predict the next token.

**Cross-entropy loss**:
The average negative log-probability that the model assigns to the correct next token.

**Foundation model**:
A pretrained model that hasn't been fine-tuned. It continues text and doesn't follow instructions.
_Avoid_: base model, chat model

**Pretrained weights**:
Parameters learned by someone else's pretraining run, here OpenAI's GPT-2 release, loaded into this repo's model.

**Model size**:
A GPT-2 variant named by its parameter count: 124M (small), 355M (medium), 774M (large) or 1558M (XL).

**Checkpoint**:
Saved model parameters, and optionally optimizer state, that can be reloaded later.

**Fine-tuning**:
Further training of a foundation model on a smaller, task-specific dataset. It is the next stage after pretraining and isn't in this repo yet.

## Generation

**Start context**:
The text you give the model to continue.
_Avoid_: seed text

**Greedy decoding**:
Always picking the token with the highest logit as the next token.

**Temperature**:
A divisor applied to logits before sampling. Values below 1 make the output more predictable and values above 1 make it more varied.

**Top-k sampling**:
Sampling the next token only from the k tokens with the highest logits.
