import numpy as np
import torch
#from utils import text2vec
import sys
import glob
import random
from PIL import Image
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
import multiprocessing


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    custum collate because the default one do not handle caption very well
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: lists of int of different length
            - caption label
    Returns:
        - tensor of shape (batch, 3, 256, 256)
        - tensor of shape (batch, max caption length)
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ps = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    length = len(captions[0])
    targets = torch.zeros(len(captions), length).long()
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = torch.LongTensor(cap[:end])  
    
          
    return images, targets, torch.tensor(ps)
    

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
    def __init__(self, filename, baseDir='./', maxLength=20, p=0.5):
        self.transform =transforms.Compose(
                            (
                            transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.ToTensor())
                            )
        self.imagesList = set()
        self.annotations = {}
        self.baseDir = baseDir
        self._readFile(filename)
        self.imagesList = list(self.imagesList)
        self.p = p
        
    def _readFile(self, filename):
        with open(filename) as f:
            for l in f:
                imageName, listVal = l.split('\t')
                
                if '#' in imageName:
                	imageName, imageNumber = imageName.split('#')
                
                listVal = [int(i) for i in listVal.split(' ') if not '\n' in i]
                
                if imageName in self.imagesList:
                    self.annotations[imageName].append(listVal)
                else:
                    self.imagesList.add(imageName)
                    self.annotations[imageName] = [listVal]

    def _openImage(self, index):
        im = Image.open(self.baseDir + self.imagesList[index])
        if not im.mode == 'RGB':
            im = im.convert("RGB")
        return self.transform(im)


    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, index):
        im = self._openImage(index)
        if random.random() > self.p: #take a positive caption
            annots = self.annotations[self.imagesList[index]]
            p = 1
            
        else: #negative caption
            c = random.randint(0, self.__len__()-1)
            while c == index:
	            c = random.randint(0, self.__len__()-1)
            annots = self.annotations[self.imagesList[c]]
            p = -1
            
        i = random.randint(0,len(annots)-1) #choose a random caption from the list of caption
        return im, torch.LongTensor(annots[i]), p

if __name__ == "__main__":
    print('Test datasets')
    dataset = AnnotatedImageDataset("../data/coco.annot", '/data/coco/train2014/')
    print("Nb images : ", len(dataset))
    print("First item : ", dataset[0])
    
    dataloader = DataLoader(dataset=dataset, batch_size=80,
                        shuffle=True, num_workers=multiprocessing.cpu_count(), drop_last=True,
                        collate_fn=collate_fn)
    
    for b, batch in enumerate(dataloader):
        print("%2.2f"% (b/len(dataloader)*100), '\%', end='\r')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
