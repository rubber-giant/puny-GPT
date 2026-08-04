from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

train = ds["train"]
val = ds["validation"]

print(train[0])
print(len(train))
