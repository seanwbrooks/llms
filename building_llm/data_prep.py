import urllib.request
import re

URL = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
# urllib.request.urlretrieve(URL, file_path)

# instead of verdict, let's use the "test" file I setup
with open("text-for-data-prep.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

# use regular expression to split on whitespace
result = re.split(r'(\s)', raw_text)
print(result)

# use regex with commas, periods
result = re.split(r'([,.]\s)', raw_text)
print(result)

# there are still redudant whitespace
result = [item for item in result if item.strip()]
print(result)

# we need to handle ALL punctuation, i.e. ":", "?", etc.
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
print(result)

# The last result is "good", we can call "preprocessing" 
preprocessed = result

# Nekt let's tokenize our preprocessed set
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
print(all_words)

# Creating a vocabulary
vocab = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
	print(item)
	if i >= 50:
		break

# Implementing a text tokenizer that includes "special" cases, unrecognized words and end of text
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

# A tokenizer that handles unknown words and end of text
class SimpleTokenizerV2:
	def __init__(self, vocab):
		self.str_to_int = vocab
		self.int_to_str = { i:s for s,i in vocab.items() }
	def encode(self, text):
		preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
		preprocessed = [
			item.strip() for item in preprocessed if item.strip()
		]
		preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

		ids = [self.str_to_int[s] for s in preprocessed]
		return ids
	def decode(self, ids):
		text = " ".join([self.int_to_str[i] for i in ids])
		text = re.sub(r'\s+([,.:;?!()\'])', r'\1', text)
		return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
encoded = tokenizer.encode(text)
print(encoded)

print(tokenizer.decode(encoded))
