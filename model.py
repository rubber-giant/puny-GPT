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

================================================================================
## Transformer 
================================================================================

class Attention_h(nn.Module):
    def __init__(self, block_size, head_size):
        super().__init__()
        self.query= nn.Linear(n_embd, head_size, bias=False)
        self.key= nn.Linear(n_embd, head_size, bias=False)
        self.value= nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask',torch.tril(torch.ones(block_size,block_size),device=device))

    def forward(self, idx):
        B,T,C = idx.shape
        q = query(idx)      #(B,T,head_size)
        k = key(idx)        #(B,T,head_size)
        v = value(idx)      #(B,T,head_size)
        wei = q @ k.transpose(-2,-1)    # Affinity Matrix
        wei = torch.mul(wei, (C**-0.5))     # Normalize values
        
        wei = wei.masked_fill(mask==0,float('-inf'))
        prob_wei = F.softmax(wei, dim=1)
        
        out = prob_wei @ v
        return out


class Transformer(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, num_head, head_size):        ## head_size * num_head = n_embd Always
        super().__init__()
            # Token lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)   #(input size,output size)
        
        self.att_head = nn.ModuleList([Attention_h(head_size) for i in range(num_head]))
   
    def forward(self,idx,targets=None):
        B,T = idx.shape
        # idx is x and targets is y. With dimension (B,T)
        x_embd = self.token_embedding_table(idx)    #(B,T,C)  C is n_embd
        x_embd = torch.mul(x_embd,torch.sqrt(n_embd/vocab_size))
        pos_embd = self.positional_embedding_table(torch.arange(T,device=device))  #(T,C) C is n_embd

        embd = torch.add(tok_embd,pos_embd)     #PE + TE    #(B,T,C)

        ## Multi-Head Attention
        out = torch.cat([head(embd) for head in self.att_head], dim=-1)

        logits = self.lm_head(out)     #(B,T, n_embd)
        
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

            # One final dice-roll(Don't want same type of answers everytime)
            idx_next = torch.multinomial(prob, num_samples=1) # (B,1)

            # Appending new values to the input again, so generation continues
            idx = torch.cat((idx, idx_next), dim=1) # dim =1 means add more columns
        return idx


#m = BigramLanguageModel(vocab_size)
#logits, loss = m(x, y)
#print(tokenizer.decode(m.generate(torch.tensor([tokenizer.encode("Hi I'am Tanmay")], dtype= torch.long), max_token_length=50)[0].tolist()))
#print(logits.shape)




