import torch
import torchvision.models
import torch.nn as nn
import torch.nn.functional as f

import numpy as np


class EncoderGRU(nn.Module):
    def __init__(self, input_size=300, hidden_size=4096):
        super(EncoderGRU, self).__init__()
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.cell = nn.GRUCell(input_size, hidden_size)

        #init vweight
        self.cell.weight_hh.data.normal_(0, 0.2)
        self.cell.weight_ih.data.normal_(0, 0.2)

    def forward(self, inp, hidden):
        #embedded = self.embedding(inp).view(1, 1, -1)
        #output = embedded
        for i in range(len(inp)):
            hidden = self.cell(inp[i], hidden)
        #hidden = f.normalize(hidden,p=2, dim=-1)
        return hidden


class EncoderEmbedding(nn.Module):
    def __init__(self):
        super(EncoderEmbedding, self).__init__()
        self.hidden_size = 2048
        self.embedding = torch.load('/workspace/imageSimilarity/embedding/embedding.save').cuda()
        self.gru = nn.GRU(300, 2048, batch_first=True)
        self.nbLayer = 1

    def forward(self, input):
        input = self.embedding(input)
        hidden = self.initHidden(input.shape[0])
        return self.gru(input, hidden.cuda())[1][0]

    def initHidden(self, batchSize):
        return torch.zeros(self.nbLayer, batchSize, self.hidden_size)


if __name__ == '__main__':
    enc = EncoderEmbedding().cuda()
    inputs = torch.LongTensor(1,20)
    for i in range(1):
        inputs[i] = torch.LongTensor(np.random.randint(9000,size=20))
    inputs = inputs.cuda()
    outs = enc(inputs)
    outs = outs.detach()
    print(outs)
