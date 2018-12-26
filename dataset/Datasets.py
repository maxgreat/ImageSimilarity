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

class AnnotatedImageDataset(ImageDataset):
    """
        Annotation must be :
            imageName#n <\t> annotation
    """
    def __init__(self, directory, annotation, transform=transforms.ToTensor()):
        super().__init__(directory, transform)
        self.annotations = readAnnotation(annotation)

    def __getitem__(self, index):
        annots = self.annotations[imagesList[index]]
        i = random.randint(0,len(annots)-1)
        return self.transform(Image.open(self.imagesList[index])), annots[i], self.imagesList[index]
