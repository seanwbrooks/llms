# Required `pip install tiktoken`
from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"))

with open("text-for-data-prep.txt", "r", encoding="UTF-8") as f:
	text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

