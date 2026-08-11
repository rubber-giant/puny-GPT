from model import get_batches
from model import BigramLanguageModel
from tokenizer import CharTokenizer as cton
import torch.nn as nn
import numpy as np
import torch

# Tokenizer class setup
tok = cton.load()
vocab_size = tok.vocab_size

# HyperParams
batch_size = 32     # how many independent sequences in parallel?
block_size = 128    # the context length ofcourse
max_iter = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Model instance
print(vocab_size)
model = BigramLanguageModel(vocab_size)
model = model.to(device)
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for _ in range(max_iter):
    # get data
    x,y = get_batches(batch_size, block_size, split='train')
    x.to(device)
    y.to(device)

    # Forward Pass
    logits, loss = model(x,y)
    
    # Backward Pass
    loss.backward()

    # Update params
    optimizer.step()

    # zero grad
    optimizer.zero_grad(set_to_none=True)

    print(_,":",loss.item())

#print(tok.decode(model.generate(torch.tensor([tok.encode("Lily lived i")], dtype= torch.long,device=device), max_token_length=500)[0].tolist()))

# Generate from the model
context = "Moon is"
max_token_length = 500
encoded = tok.encode(context)
tensor = torch.tensor([encoded], dtype=torch.long, device=device)
generate = model.generate(tensor, max_token_length)
decoded = tok.decode(generate[0].tolist())
print(decoded)

