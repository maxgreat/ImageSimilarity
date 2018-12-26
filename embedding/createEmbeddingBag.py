import fastText
import torch
import torch.nn as nn
import numpy as np
import io
import os

def getFileSize(f):
    old_file_position = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(old_file_position, os.SEEK_SET)
    return size

def getLines(f):
    old_file_position = f.tell()
    for i, line in enumerate(f):
        pass
    f.seek(old_file_position, os.SEEK_SET)
    return i+1

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = [int(i) for i in fin.readline().split()]
    tokens = fin.readline().rstrip().split(' ')
    vocab = {tokens[0]:0}
    print("Checking file size .... ")
    s = getLines(fin)
    matrix = [[]]*s
    matrix[0] = [float(t) for t in tokens[1:]]
    print('Reading vectors')
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        matrix[i] = [float(t) for t in tokens[1:]]
        vocab[tokens[0]] = i+1
    return vocab, np.array(matrix)

vocab, matrix = load_vectors('/data/wiki-full.vec')
print('Matrix size:', matrix.shape)
embedBags = nn.Embedding(matrix.shape[0], matrix.shape[1],padding_idx=0, sparse=False, max_norm=1.0)
embedBags.weight.data.copy_(torch.Tensor(matrix))

text = 'this is a test'

tokens = fastText.tokenize(text)

input = torch.LongTensor(1, len(tokens))
input[0] = torch.LongTensor([vocab[i] for i in tokens])

bag = embedBags(input).detach()

print(bag)
torch.save(vocab, "vocab.save")
torch.save(embedBags,'embedding.save')
