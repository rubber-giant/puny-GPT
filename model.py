from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch

def get_data(split):
    data = np.memmap(
            "data/train.bin" if split=='train' else "data/val.bin",
            dtype = np.uint8,
            mode = 'r'
            )
    return data

# Helper function
def get_batches(batch_size, block_size, split='val'):
    data = get_data(split)
    ix= torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([
            torch.from_numpy(data[i:i+block_size].copy()).long()
            for i in ix]
            )
    y = torch.stack([
            torch.from_numpy(data[i+1:i+1+block_size].copy()).long()
                              for i in ix]
        )
    x = torch.squeeze(x)
    y = torch.squeeze(y)
    return x, y

# mean business
batch_size = 4
block_size = 8

vocab_size = 175
# DataLoader
x,y = get_batches(batch_size, block_size, split='train')
print("X:",x)
print("Y:",y)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
            # Token lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
   
    def forward(self,idx,targets):

        # idx is x and targets is y. With dimension (B,T)
        logits = self.token_embedding_table(idx)    #(B,T,C)
        
        # Cross_entropy takes in (B,C) as input, and (B) as targets
        # Therefore we flatten our logits and targets 
        # It makes sense just visit torch's website
        B, T, C = logits.shape

        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

m = BigramLanguageModel(vocab_size)
logits, loss = m(x, y)
print(loss)





