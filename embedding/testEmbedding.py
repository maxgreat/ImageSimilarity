import torch
import time
import fastText

start = time.clock()
embed = torch.load('embedding.save')
vocab = torch.load('vocab.save')

batch = 3
text = 'this is a test'

tokens = fastText.tokenize(text)
inputs = torch.LongTensor(10, len(tokens))

for i in range(10):
    inputs[i] = torch.Tensor([vocab[t] for t in tokens])

bag = embed(inputs).detach()

print(bag.shape)
