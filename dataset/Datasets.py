import numpy as np
import torch
#from utils import text2vec
import sys
import glob
import random
from PIL import Image
import torchvision.transforms as transforms
import random




class ImageDataset(torch.utils.data.Dataset):
    """
        Receive directory with images inside, load images
    """
    def __init__(self, directory, transform=transforms.ToTensor()):
        super().__init__()
        self.dir = directory
        self.imagesList = glob.glob(directory+'*.jpg')
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(Image.open(self.imagesList[index])), self.imagesList[index]

    def __len__(self):
        return len(self.imagesList)


def readAnnotation(filename):
    with open(filename) as f:
        annot = {}
        for line in f:
            name, annotation = line.split('\t')
            if '#' in name :
                name = name.split('#')[0]
            if name in annot:
                annot[name].append(annotation)
            else:
                annot[name] = [annotation]

class AnnotatedImageDataset(torch.utils.data.Dataset):
    """
        Annotation must be :
            imageName#<imageNumber><\t><list of int>
    """
    def __init__(self, filename, baseDir='./', transform=transforms.ToTensor()):
        self.transform = transform
        self.imagesList = set()
        self.annotations = {}
        self.baseDir = baseDir
        self.readFile(filename)
        self.imagesList = list(self.imagesList)
    def readFile(self, filename):
        with open(filename) as f:
            for l in f:
                imageName, listVal = l.split('\t')
                imageName, imageNumber = imageName.split('#')
                listVal = listVal.split(' ')
                if imageName in self.imagesList:
                    self.annotations[imageName].append(listVal)
                else:
                    self.imagesList.add(imageName)
                    self.annotations[imageName] = [listVal]

    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, index):
        annots = self.annotations[self.imagesList[index]]
        i = random.randint(0,len(annots)-1)
        return self.transform(Image.open(self.baseDir+self.imagesList[index])), annots[i], self.imagesList[index]


if __name__ == "__main__":
    print('Test datasets')
    dataset = AnnotatedImageDataset("/data/flickr30k/results_20130124.token", '/data/flickr30k/flickr30k_images/')
    print("Nb images : ", len(dataset))
    print("First item : ", dataset[0])
