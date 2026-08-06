from datasets import load_dataset
from tokenizer import CharTokenizer
import json
import numpy as np

# load dataset
ds = load_dataset("roneneldan/TinyStories")

train = ds["train"]
val = ds["validation"]

## Vocab setup
char = set()
for story in train['text']:
    char.update(story)
vocab = sorted(char)
vocab_size = len(char)

# Save vocab separately 
with open("data/vocab.json",'w') as f:
    json.dump(vocab,f)

## Tokenizer
tokenizer = CharTokenizer(vocab)

# encode train data
with open("data/train.bin","wb") as f:
   for story in train["text"]:
       token = tokenizer.encode(story)
        np.array(token,
                 dtype = np.uint8
                ).tofile(f)    


# encode val data
with open("data/val.bin", 'wb') as f:
    for story in val['text']:
        token = tokenizer.decode(story)
        np.array(token,
                 dtype = np.uint8
                 ).tofile(f)

