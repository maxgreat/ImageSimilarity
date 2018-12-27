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
        Annotation file must be :
            imageName<#imageNumber> <\t> <list of int>
            
        getitem returns :
        	the open image, 
        	one random caption,
        	1 image the caption correspond to the image, -1 otherwise
        	
       	The probability to get negative caption is given by p (default 0.5)
        	
    """
    def __init__(self, filename, baseDir='./', maxLength=20, p=0.5, transform=transforms.ToTensor()):
        self.transform = transform
        self.imagesList = set()
        self.annotations = {}
        self.baseDir = baseDir
        self.readFile(filename)
        self.imagesList = list(self.imagesList)
        self.p = p
        
    def readFile(self, filename):
        with open(filename) as f:
            for l in f:
                imageName, listVal = l.split('\t')
                
                if '#' in imageName:
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
        if random.random() > self.p: #take a positive caption
            annots = self.annotations[self.imagesList[index]]
            i = random.randint(0,len(annots)-1) #choose a random caption from the list of caption
            image = Image.open(self.baseDir+self.imagesList[index])
            return self.transform(image), annots[i], 1
        else: #negative caption
            c = random.randint(0, self.__len__()-1)
            while c == index:
	            c = random.randint(0, self.__len__()-1)
            annots = self.annotations[self.imagesList[c]]
            i = random.randint(0,len(annots)-1) #choose a random caption from the list of caption
            image = Image.open(self.baseDir+self.imagesList[index])
            return self.transform(image), annots[i], -1

if __name__ == "__main__":
    print('Test datasets')
    dataset = AnnotatedImageDataset("../data/coco.annot", '/data/coco/train2014/')
    print("Nb images : ", len(dataset))
    print("First item : ", dataset[0])
