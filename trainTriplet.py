import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader

from models import models
from models import text_model

from dataset import Datasets
from dataset.flickr30k import flickDataset, TripletDataset

import argparse
import sys
import multiprocessing
import numpy as np

from utils.loss import EuclideanLoss, CosineLoss, EuclideanTriple

from tensorboardX import SummaryWriter




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
    writer = SummaryWriter()
    score = testModel(model, dataloaderTest, batchSize, vocab)
    writer.add_scalar('data/score', score, 0)
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
                writer.add_scalar('data/loss', running_loss, b+(epoch*len(dataset)))
                print('[%d, %5d] loss: %.3f' % (epoch+1, b+1, running_loss / 20))
                running_loss = 0.0

        torch.save(model, save_output+'tripletLast.save')
        nScore = testModel(model, dataloaderTest)
        writer.add_scalar('data/score', score, epoch*len(dataset))
        if nScore > score :
            score = nScore
            print('Saving best model')
            torch.save(model, save_output+'bestTriplet.save')
            
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help="optional - directory to save file - default data/",
                        default='data/')
    parser.add_argument('-e', '--epoch', help="optional - number of epoch to do - default 5",
                        type=int, default=5)
    parser.add_argument('-b', '--batchSize', help="optional - batchSize - default 32",
                        type=int, default=128)
    parser.add_argument('-r', '--resume', help="optional - resume training with the given filename")
    parser.add_argument('-l', '--learningrate', help="optional - learning rate value - default 0.01",
                        type=float, default=0.01)
    parser.add_argument('-d', '--dataset', help="optional - file with dataset",
                        default="data/coco.annot")
    parser.add_argument('--baseDir', help="optional - directory for images",
                        default="/data/coco/train2014/")
    parser.add_argument('--triplet', help="optional - if set, use  triplets", default='False')
    args = parser.parse_args()


    
    #define train dataset and dataloader
    dataset = Datasets.AnnotatedImageDataset(args.dataset, args.baseDir)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchSize,
                        shuffle=True, num_workers=multiprocessing.cpu_count(), drop_last=True,
                        collate_fn=Datasets.collate_fn)
    
    #define test dataset
    datasetTest = Datasets.AnnotatedImageDataset("data/test.annot", baseDir="/data/flickr30k/flickr30k_images/", p=0)
    dataloaderTest = DataLoader(dataset=datasetTest, batch_size=args.batchSize,
                        shuffle=False, num_workers=multiprocessing.cpu_count(), drop_last=False, 
                        collate_fn=Datasets.collate_fn)


    if args.triplet == 'False':
        print('Train with couples')
        if args.resume:
            model = torch.load(args.resume)
        else:
            print('Creating new model')
            model = models.DoubleNet(models.resnetExtraction(),
                                    text_model.EncoderEmbedding())
        train(model=model, save_output=args.output,
                nbepoch=args.epoch,
                learningrate=args.learningrate, dataloader=dataloader, dataloaderTest=dataloaderTest)
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
                nbepoch=args.epoch,
                learningrate=args.learningrate,vocab=torch.load(args.vocab),
                datasetFile=args.dataset, datasetDir=args.baseDir)

