from torch.nn import functional as F
#from tokenizer import CharTokenizer as cton
import torch.nn as nn
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    x = torch.squeeze(x).to(device)
    y = torch.squeeze(y).to(device)
    return x, y

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
            # Token lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
   
    def forward(self,idx,targets=None):

        # idx is x and targets is y. With dimension (B,T)
        logits = self.token_embedding_table(idx)    #(B,T,C)
        
        # For inference and train separation in loop
        if targets==None:
            loss=None
     

        else:
            # Cross_entropy takes in (B,C) as input, and (B) as targets
            # Therefore we flatten our logits and targets 
            # It makes sense just visit torch's website
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_token_length):
        for _ in range(max_token_length):
            # Calculate logits for our input
            logits, loss = self(idx)

            # Pick the last token's/Timestep's logits only, since it's a bigram
            logits = logits[:,-1,:]   # all batches, last token, all channels (B,C)
            
            # Create it's probability distribution(Softmaxxing)
            prob = F.softmax(logits, dim =1)

            # One final diceroll(Don't want same type of answers everytime)
            idx_next = torch.multinomial(prob, num_samples=1) # (B,1)

            # Appending new values to the input again, so generation continues
            idx = torch.cat((idx, idx_next), dim=1) # dim =1 means add more columns
        return idx


#m = BigramLanguageModel(vocab_size)
#logits, loss = m(x, y)
#print(tokenizer.decode(m.generate(torch.tensor([tokenizer.encode("Hi I'am Tanmay")], dtype= torch.long), max_token_length=50)[0].tolist()))
#print(logits.shape)




