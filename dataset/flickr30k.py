import numpy as np
import torch
#from utils import text2vec
import sys
import random
from PIL import Image
import torchvision.transforms as transforms
import fastText
from torch.utils.data import DataLoader



class flickDataset(torch.utils.data.Dataset):
    def __init__(self, filename, baseDir='/data/flickr30k/flickr30k_images/', proba_neg=0.5):
        #self.captions = {}
        self.imList = set()
        self.captionsText = {}
        #self.text2vec = text2vec.Text2Vec()
        self.transform = transforms.Compose(
                            (
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.ToTensor())
                            )
        self.baseDir = baseDir
        self.proba = proba_neg

        with open(filename) as f:
            print('Reading  file', filename)
            for line in f:
                if not '\t' in line:
                    continue
                name, caption = line.split('\t')
                name = name.split('#')[0]
                self.imList.add(name)
                if name in self.captionsText:
                    #self.captions[name].append(self.text2vec.encodeText(caption))
                    self.captionsText[name].append(caption)
                else:
                    #self.captions[name] = [self.text2vec.encodeText(caption)]
                    self.captionsText[name] = [caption]
        self.imList = np.array(list(self.imList))
        batch = 0

    def countCaptions(self):
        l = 0
        for im in self.captionsText:
            l += len(self.captionsText[im])
        return l

    def openImage(self, index):
        im = Image.open(self.baseDir + self.imList[index])
        if not im.mode == 'RGB':
            im = im.convert("RGB")
        return self.transform(im)


    def __getitem__(self, index):
        if random.random() > self.proba: #positive example
            capts = self.captionsText[self.imList[index]]
            p = 1
        else: # negative example
            c = random.randint(0, self.__len__()-1)
            while c == index:
                c = random.randint(0, self.__len__()-1)
            capts = self.captionsText[self.imList[c]]
            p = -1
        i = random.randint(0,len(capts)-1)
        return self.openImage(index), capts[i], p

    def __len__(self):
        return len(self.imList)

    def shuffle():
        np.random.shuffle(self.imList)



class TripletDataset(flickDataset):
    def __init__(self, filename, baseDir='/data/flickr30k/flickr30k_images/'):
        super().__init__(filename, baseDir)

    def __getitem__(self, index):
        posCapts = self.captionsText[self.imList[index]]
        c = random.randint(0, self.__len__()-1)
        while c == index:
            c = random.randint(0, self.__len__()-1)

        negCapts = self.captionsText[self.imList[c]]
        return self.openImage(index), posCapts[random.randint(0,len(posCapts)-1)], negCapts[random.randint(0,len(negCapts)-1)]


if __name__ == "__main__":
    print('Test flicker dataset')
    dataset = flickDataset("/data/flickr30k/results_20130124.token")
    print("Nb images : ", len(dataset.imList))
    print("Nb captions :", dataset.countCaptions())
    print("First item : ", dataset[0])
    dataset = flickDataset("/data/coco/annotation/captions.txt", baseDir='/data/coco/train2014/', proba_neg=0)
    print("Nb images : ", len(dataset.imList))
    print("Nb captions :", dataset.countCaptions())
    print("First item : ", dataset[0])
    dataset = TripletDataset("/data/coco/annotation/captions.txt", baseDir='/data/coco/train2014/')
    print("Nb images : ", len(dataset.imList))
    print("Nb captions :", dataset.countCaptions())
    print("First item : ", dataset[0])
