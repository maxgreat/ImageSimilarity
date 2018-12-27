import numpy as np
import torch
#from utils import text2vec
import sys
import glob
import random
from PIL import Image
import torchvision.transforms as transforms
import random
import json



def createFile(filename, fout):
    """
        Create a formated file for coco
        inout : json file with 'images' and 'annotations' categories
    """
    l = json.load(open(filename))
    annotations = l['annotations']
    images = l['images']
    ids = {}
    countImages = {}
    for im in images:
        ids[im['id']] = im['file_name']

    with open(fout,'w') as fout:
        for i, annot in enumerate(annotations):
            print("%2.2f"% (i/len(annotations)*100), '\%', end='\r')
            id = annot['image_id']
            fout.write(ids[id]+'\t'+annot['caption']+'\n')



class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, directory, annotationFile, transform=transforms.ToTensor()):
        super().__init__()


    def __len__(self):
        return len(self.images.keys())


if __name__ == '__main__':
    createFile('/data/coco/annotation/captions_train2014.json', '/data/coco/annotation/captions.txt')
