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


def testModel(model, dataloader):
    model = model.eval()
    for b, batch in enumerate(dataloader):
        output = model(batch[0].cuda(), batch[1].cuda())
        labels = batch[2].type(torch.float32).cuda()

        nbCorrect = 0
        pdist = nn.PairwiseDistance()
        sim = pdist(output[0], output[1])
        #sim = f.cosine_similarity(output[0], output[1], -1)
        for i, s in enumerate(sim):
            if labels[i] > 0:
                if s <= 1:
                    nbCorrect+=1
                print('Similar distance :', float(s.detach()))
            else:
                if s > 1:
                    nbCorrect += 1
                print('Different distance :', float(s.detach()))
    print('Correct : ', nbCorrect, '/', len(sim))
    model = model.train()
    return nbCorrect





def train(model, save_output, nbepoch, learningrate, dataloader, dataloaderTest):
    """
        Train a given model for nbepoch epochs on the given dataset
    """
    model = model.train().cuda()

    criterion=EuclideanLoss(1).cuda()
    #criterion=CosineLoss(0.5).cuda()

    optimizer=optim.Adam([
                    {
                    'params': model.net2.gru.parameters(),
                    'params': model.net1.module.layer4.downsample.parameters()
                        },
                    {'params': model.net1.module.parameters(),
                    'params':model.net2.embedding.parameters(),
                    'params': model.net1.module.layer4.parameters(), 'lr': 0.0}
                ], lr=0.01)

    running_loss = 0

    #model = nn.DataParallel(model)
    writer = SummaryWriter()
    score = testModel(model, dataloaderTest)
    writer.add_scalar('data/score', score, 0)
    for epoch in range(nbepoch):
        print("Epoch ", epoch, "/", nbepoch)
        for b, batch in enumerate(dataloader):
            if b%2 == 1:
                print("%2.2f"% (b/len(dataloader)*100), '\%', end='\r')

            #batch contain couples of (image,caption)
            output = model(batch[0].cuda(), batch[1].cuda())

            optimizer.zero_grad()
            labels = batch[2].type(torch.float32).cuda()
            loss = criterion(output[0], output[1], labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if b % 10 == 9: # print every 10 mini-batches
                writer.add_scalar('data/loss', running_loss, b+(epoch*len(dataset)))
                print('[%d, %5d] loss: %.3f' % (epoch+1, b+1, running_loss / 20))
                running_loss = 0.0

        torch.save(model, save_output+'last.save')
        nScore = testModel(model, dataloaderTest)
        writer.add_scalar('data/score', score, epoch*len(dataset))
        if nScore > score :
            score = nScore
            print('Saving best model')
            torch.save(model, save_output+'best.save')
            
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


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

    if args.resume:
        model = torch.load(args.resume)
    else:
        print('Creating new model')
        model = models.DoubleNet(models.resnetExtraction(),
                                text_model.EncoderEmbedding())
    train(model=model, save_output=args.output,
            nbepoch=args.epoch,
            learningrate=args.learningrate, dataloader=dataloader, dataloaderTest=dataloaderTest)
