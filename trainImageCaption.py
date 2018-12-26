import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader

from models import models
from models import text_model

from dataset.flickr30k import flickDataset, TripletDataset

import argparse
import sys
import multiprocessing
import numpy as np


class EuclideanLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin=margin

    def forward(self,x,y,p):
        s = 0
        pdist = nn.PairwiseDistance()
        dists = pdist(x,y)
        for i, eq in enumerate(p):
            if eq == 1:
                s += dists[i]
            else:
                s+= max(0, self.margin - dists[i])
        return s

class CosineLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin=margin

    def forward(self,x,y,p):
        s = 0
        dists = f.cosine_similarity(x,y)
        for i, eq in enumerate(p):
            if eq == 1:
                s += 1-dists[i] #equal means == 1
            else:
                s+= max(0, dists[i] - self.margin) #similarity must be smaller than margin
        return s

class EuclideanTriple(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin=margin

    def forward(self,x,y,z):
        s = 0
        pdist = nn.PairwiseDistance()
        distsP = pdist(x,y)
        distsN = pdist(x,z)
        for i,d in enumerate(distsP):
            s = max(0, d + self.margin - distsN[i])
        return s


def testModel(model, dataloader, batchSize, vocab):
    model = model.eval()
    for b, batch in enumerate(dataloader):
        input1 = batch[0].cuda()

        sentences = batch[1]
        input2 = torch.LongTensor(len(sentences), 20)
        for i, s in enumerate(sentences):
            for j in range(20):
                if j < len(s) and s[j] in vocab:
                    input2[i][j] = vocab[s[j]]
                else:
                     input2[i][j] = 0
        input2 = input2.cuda()

        output = model(input1, input2, input2)
        labels = batch[2].type(torch.float32).cuda()

        nbCorrect = 0
        pdist = nn.PairwiseDistance()
        sim = pdist(output[0], output[1])
        sim2 = pdist(output[1], output[2])
        print(sim2)
        #sim = f.cosine_similarity(output[0], output[1], -1)
        for i, s in enumerate(sim):
            if labels[i] > 0:
                if s <= 0.5:
                    nbCorrect+=1
                print('Similar distance :', float(s.detach()))
            else:
                if s > 0.5:
                    nbCorrect += 1
                print('Different distance :', float(s.detach()))
    print('Correct : ', nbCorrect, '/', len(sim))
    model = model.train()
    return nbCorrect


def trainTriplet(model, save_output, nbepoch, batchSize, learningrate, vocab, datasetFile, datasetDir):
    dataset = TripletDataset(datasetFile, datasetDir)
    dataloader = DataLoader(dataset=dataset, batch_size=batchSize,
                        shuffle=True, num_workers=multiprocessing.cpu_count(), drop_last=True)
    datasetTest = flickDataset("/data/flickr30k/test.tokens", proba_neg=0)
    dataloaderTest = DataLoader(dataset=datasetTest, batch_size=batchSize,
                        shuffle=False, num_workers=multiprocessing.cpu_count(), drop_last=False)
    model = model.cuda()
    #criterion = nn.CosineEmbeddingLoss(margin=0.5)
    criterion=EuclideanTriple(0.5).cuda()
    #
    # optimizer=optim.SGD([
    #                 {
    #                 'params': model.net2.gru.parameters(),
    #                     },
    #                 {'params': model.net1.module.parameters(), 'params':model.net2.embedding.parameters(), 'lr': 0.0}
    #             ], lr=learningrate)
    #
    #

    optimizer=optim.Adam([
                    {
                    'params': model.net2.gru.parameters(),

                        },
                    {'params': model.net1.module.parameters(),
                    'params':model.net2.embedding.parameters(), 'lr': 0.001}
                ], lr=0.005)

    model = model.train()
    running_loss = 0

    #model = nn.DataParallel(model)
    #score=0
    score = testModel(model, dataloaderTest, batchSize, vocab)
    for epoch in range(nbepoch):
        print("Epoch ", epoch, "/", nbepoch)
        for b, batch in enumerate(dataloader):
            if b%2 == 1:
                print("%2.2f"% (b/len(dataloader)*100), '\%', end='\r')
            #batch contain couples of (image,caption)
            #inputs = torch.Tensor()
            input1 = batch[0].cuda()

            sentences = batch[1]
            input2 = torch.LongTensor(len(sentences), 20)
            for i, s in enumerate(sentences):
                for j in range(20):
                    if j < len(s) and s[j] in vocab:
                        input2[i][j] = vocab[s[j]]
                    else:
                         input2[i][j] = 0
            input2 = input2.cuda()

            sentences = batch[2]
            input3 = torch.LongTensor(len(sentences), 20)
            for i, s in enumerate(sentences):
                for j in range(20):
                    if j < len(s) and s[j] in vocab:
                        input3[i][j] = vocab[s[j]]
                    else:
                         input3[i][j] = 0
            input3 = input3.cuda()

            output = model(input1, input2, input3)
            optimizer.zero_grad()
            loss = criterion(output[0], output[1], output[2])
            if not loss == 0:
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]

            if b % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, b+1, running_loss / 20))
                running_loss = 0.0

        torch.save(model, save_output+'tripletLast.save')
        nScore = testModel(model, dataloaderTest, batchSize, vocab)
        if nScore > score :
            score = nScore
            print('Saving best model')
            torch.save(model, save_output+'bestTriplet.save')



def train(model, save_output, nbepoch, batchSize, learningrate, vocab, datasetFile, datasetDir):
    dataset = flickDataset(datasetFile, datasetDir)
    dataloader = DataLoader(dataset=dataset, batch_size=batchSize,
                        shuffle=True, num_workers=multiprocessing.cpu_count(), drop_last=True)
    datasetTest = flickDataset("/data/flickr30k/test.tokens", proba_neg=0)
    dataloaderTest = DataLoader(dataset=datasetTest, batch_size=batchSize,
                        shuffle=False, num_workers=multiprocessing.cpu_count(), drop_last=False)
    model = model.cuda()
    #criterion = nn.CosineEmbeddingLoss(margin=0.5)
    criterion=EuclideanLoss().cuda()
    #
    # optimizer=optim.SGD([
    #                 {
    #                 'params': model.net2.gru.parameters(),
    #                     },
    #                 {'params': model.net1.module.parameters(), 'params':model.net2.embedding.parameters(), 'lr': 0.0}
    #             ], lr=learningrate)
    #
    #

    optimizer=optim.Adam([
                    {
                    'params': model.net2.module.gru.parameters(),
                    'params': model.net1.module.layer4.parameters(),
                        },
                    {'params': model.net1.module.parameters(),
                    'params':model.net2.module.embedding.parameters(), 'lr': 0.0001}
                ], lr=0.0005)


    #
    # optimizer=optim.SGD([
    #             {'params': model.net1.module.parameters(),
    #             'params': model.net2.parameters(),
    #                 }
    #         ], lr=learningrate)

    #
    # optimizer=optim.Adam(
    #                 [{'params': model.parameters()}], lr=0.0001)

    model = model.train()
    running_loss = 0

    #model = nn.DataParallel(model)
    score=0
    #score = testModel(model, dataloaderTest, batchSize, vocab)
    for epoch in range(nbepoch):
        print("Epoch ", epoch, "/", nbepoch)
        for b, batch in enumerate(dataloader):
            if b%2 == 1:
                print("%2.2f"% (b/len(dataloader)*100), '\%', end='\r')

            #batch contain couples of (image,caption)
            #inputs = torch.Tensor()
            input1 = batch[0].cuda()

            sentences = batch[1]
            input2 = torch.LongTensor(len(sentences), 20)
            for i, s in enumerate(sentences):
                for j in range(20):
                    if j < len(s) and s[j] in vocab:
                        input2[i][j] = vocab[s[j]]
                    else:
                         input2[i][j] = 0
            input2 = input2.cuda()


            output = model(input1, input2)

            optimizer.zero_grad()
            labels = batch[2].type(torch.float32).cuda()
            loss = criterion(output[0], output[1], labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if b % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, b+1, running_loss / 20))
                running_loss = 0.0

        torch.save(model, save_output+'last.save')
        nScore = testModel(model, dataloaderTest, batchSize, vocab)
        if nScore > score :
            score = nScore
            print('Saving best model')
            torch.save(model, save_output+'best.save')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help="optional - directory to save file - default /data/",
                        default='/data/')
    parser.add_argument('-e', '--epoch', help="optional - number of epoch to do - default 5",
                        type=int, default=5)
    parser.add_argument('-b', '--batchSize', help="optional - batchSize - default 32",
                        type=int, default=32)
    parser.add_argument('-r', '--resume', help="optional - resume training with the given filename")
    parser.add_argument('-l', '--learningrate', help="optional - learning rate value - default 0.01",
                        type=float, default=0.01)
    parser.add_argument('-v', '--vocab', help="optional - file with vocabulary - default vocab.save",
                        default='vocab.save')
    parser.add_argument('-d', '--dataset', help="optional - file with dataset",
                        default="/data/flickr30k/results_20130124.token")
    parser.add_argument('--baseDir', help="optional - directory for images",
                        default="/data/flickr30k/flickr30k_images/")
    parser.add_argument('--triplet', help="optional - if set, use  triplets", default='False')
    args = parser.parse_args()


    if args.triplet == 'False':
        print('Train with couples')
        if args.resume:
            model = torch.load(args.resume)
        else:
            print('Creating new model')
            model = models.DoubleNet(models.resnetExtraction(),
                                    text_model.EncoderEmbedding())
        train(model=model, save_output=args.output,
                nbepoch=args.epoch, batchSize=args.batchSize,
                learningrate=args.learningrate,vocab=torch.load(args.vocab),
                datasetFile=args.dataset, datasetDir=args.baseDir)
    else:
        print('Train with tripletLast')
        if args.resume:
            modelDouble = torch.load(args.resume)
            model = models.TripletNet(models.resnetExtraction(),
                                    text_model.EncoderEmbedding())
            model.LoadFromDouble(modelDouble)
        else:
            print('Creating new model')
            model = models.TripletNet(models.resnetExtraction(),
                                    text_model.EncoderEmbedding())
        trainTriplet(model=model, save_output=args.output,
                nbepoch=args.epoch, batchSize=args.batchSize,
                learningrate=args.learningrate,vocab=torch.load(args.vocab),
                datasetFile=args.dataset, datasetDir=args.baseDir)
