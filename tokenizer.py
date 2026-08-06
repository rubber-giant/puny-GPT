import json
class CharTokenizer:
    def __init__(self, vocab):
        self.stoi = {ch:i for i, ch in enumerate(vocab)}
        self.itos = {i:ch for i, ch in enumerate(vocab)}
    
    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    def decode(self,tokens):
        return [self.itos[i] for i in tokens]

    @classmethod
    def load(cls, path):
        with open("data/vocab.json",'r') as f:
            vocab = json.load(f)
        return cls(vocab)
