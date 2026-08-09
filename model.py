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
block_size = 128
x,y = get_batches(batch_size, block_size, split='train')
print("X:",x)
print("Y:",y)
for b in range(batch_size):
    for t in range(block_size):
        context = x[b, :t+1]
        target = y[b, t]
        print(f"when context is {context}, then target is {target}")

