# Building an LLM (from scratch)

My working code from Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) ([official repo](https://github.com/rasbt/LLMs-from-scratch)). It goes from raw text to a GPT-2-style model with a pretraining pipeline pointed at *Infinite Jest*. After that, I loaded OpenAI's real GPT-2 weights (124M → 1558M) into the same architecture.

**Progress:** chapters 2–5 are done, plus the Appendix A PyTorch primer. Next is chapter 6 (fine-tuning for classification).

For vocabulary, see [GLOSSARY.md](./GLOSSARY.md).

---

## The big picture

Every file in this repo is one stage of this pipeline:

```
 text ──► tokenizer ──► token IDs ──► token + positional embeddings
                                                │
                                                ▼
                          ┌──── Transformer block × n_layers ────┐
                          │  LayerNorm → causal multi-head attn  │
                          │  + shortcut                          │
                          │  LayerNorm → feed-forward (GELU)     │
                          │  + shortcut                          │
                          └──────────────────────────────────────┘
                                                │
                                                ▼
                         final LayerNorm ──► output head ──► logits
                                                │
             training:   cross-entropy vs. next-token targets
             generation: pick/sample next token, append, repeat
```

| Stage | Book | File |
|---|---|---|
| PyTorch basics: tensors, autograd, training loop, MPS vs CPU | App. A | `building_llm/deep_learning/` |
| Simple regex tokenizer and vocabulary, `<\|unk\|>` / `<\|endoftext\|>` | Ch 2 | `building_llm/data_prep.py` |
| BPE tokenizer (tiktoken) and sliding-window dataset/dataloader | Ch 2 | `building_llm/byte_pair_encoding.py` |
| Self-attention → causal attention → multi-head attention | Ch 3 | `building_llm/attention_mechanisms.py` |
| LayerNorm, GELU, feed-forward, shortcuts, `TransformerBlock`, `GPTModel`, greedy generation | Ch 4 | `foundation_model/generative_pretrained_transformer_model.py` |
| Loss, train/validation split, training loop, temperature and top-k, saving checkpoints | Ch 5.1–5.4 | `foundation_model/pretraining_on_unlabeled_data.py` |
| Reload a saved model and optimizer to keep training | Ch 5.4 | `foundation_model/load_saved_model.py` |
| Download GPT-2 weights and map them into `GPTModel` | Ch 5.5 | `foundation_model/retreive_gpt.py` → `gpt_download.py` → `using_gpt2_downloaded_model.py` |
| Side experiment: token count of *Infinite Jest* | — | `tokenizer.py` |

The scripts are book walkthroughs, not a library. Each one runs top to bottom and prints its intermediate results. **Importing a module runs all of its demo code too.** For example, importing `GPTModel` also runs the attention demos.

---

## Running things

There's no `requirements.txt`. The scripts were last run with Python 3.13, `torch` 2.9.1, `tiktoken` and `tensorflow` 2.20. TensorFlow is only needed to read the GPT-2 checkpoints. The scripts use Apple's `mps` device when it's available.

Scripts expect **different working directories**:

| Script | Run from | Command |
|---|---|---|
| `building_llm/byte_pair_encoding.py`, `tokenizer.py` | repo root | `python building_llm/byte_pair_encoding.py` |
| `building_llm/data_prep.py` | `text_files/` | `cd text_files && python ../building_llm/data_prep.py` |
| `foundation_model/pretraining_on_unlabeled_data.py` | repo root, as a module | `python -m foundation_model.pretraining_on_unlabeled_data` |
| `foundation_model/using_gpt2_downloaded_model.py` | `foundation_model/` | `cd foundation_model && python using_gpt2_downloaded_model.py` |
| `foundation_model/load_saved_model.py` | repo root, as a module | `python -m foundation_model.load_saved_model`. Loads `model_and_optimizer.pth` from the repo root, where the pretraining script saves it; copy it out of `saved_models/` first. Importing the pretraining module still runs its demos and loss evaluation, but no longer saves anything. |

To switch GPT-2 sizes, change `model_size` (the weights folder) and `model_name` (the config key) together at the top of `using_gpt2_downloaded_model.py`.

## Data & weights (not in git)

| Path | What | How to get it back |
|---|---|---|
| `text_files/` | `infinite-jest.txt` (3.2 MB, the pretraining corpus), a Whitman poem, and a small sample text | committed |
| `foundation_model/gpt/{124M,355M,1558M}/` | OpenAI GPT-2 TensorFlow checkpoints (0.5 GB / 1.4 GB / 6.2 GB) | `download_and_load_gpt2(model_size="124M", models_dir="gpt")` from `gpt_download.py` |
| `saved_models/` | `model.pth` and `model_and_optimizer.pth` from the 124M-config pretraining script (~650 MB each). **Untrained** (see Known issues) | rerun the pretraining script |

---

## Known issues (found while writing this README)

- **The training call is commented out** in `pretraining_on_unlabeled_data.py`, so the checkpoints it saves are **untrained** (random weights). This includes the ones in `saved_models/`: the AdamW state in `model_and_optimizer.pth` is empty, which means no optimizer step ever ran. To get an *Infinite Jest* model, uncomment `train_model_simple(...)` and rerun.
- **`tokenizer.py` measures token count, not vocabulary size**, even though its commit message says otherwise. *Infinite Jest* is 794,776 tokens.
- **`llm_exercise_a1.ipynb`** is a notebook with one code cell, and that cell contains the raw JSON of a different notebook (Raschka's STAT 453 course). It looks like a mis-paste.

---

## Where to pick up

1. Re-read chapter 5's summary. Then run `using_gpt2_downloaded_model.py` with 124M to make sure everything still works.
2. **Chapter 6:** fine-tune for classification (spam vs. not spam). Swap the output head for a 2-class layer and train only the last blocks.
3. **Chapter 7:** instruction fine-tuning. This is the step that makes the model follow prompts like the "list of 10 random words" request in the GPT-2 script, which a foundation model can only continue as text.
