from datasets import load_dataset

# load dataset
ds = load_dataset("roneneldan/TinyStories")

train = ds["train"]
val = ds["validation"]

# Vocab setup
char = sorted(set(''.join(train['text'])))
vocab_size = len(char)

## Tokenize
stoi = {ch:i for i,ch in enumerate(char)}
itos = {i:ch for i,ch in enumerate(char)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(i):
    return (''.join(itos[i] for i in i))

print(encode("hello babygirl"))
print(decode(encode("hello babygirl")))
