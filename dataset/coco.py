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
        input : 
        	json file with 'images' and 'annotations' categories (from coco download)
        output : 
        	text file with each line in the form : <image name> <caption>
    """
    l = json.load(open(filename))
    annotations = l['annotations']
    images = l['images']
    ids = {} # dictionary image index -> image name
    countImages = {}
    for im in images:
        ids[im['id']] = im['file_name']

    with open(fout,'w') as fout:
        for i, annot in enumerate(annotations):
            print("%2.2f"% (i/len(annotations)*100), '\%', end='\r')
            idImage = annot['image_id']
            
            # check for error in text
            if '\n' in annot['caption']:
            	newCaption = annot['caption'].replace('\n','')
            	fout.write(ids[idImage]+'\t'+newCaption+'\n')
            else:           
            	fout.write(ids[idImage]+'\t'+annot['caption']+'\n')



if __name__ == '__main__':
    createFile('/data/coco/annotation/captions_train2014.json', '/data/coco/annotation/captionsCorrect.txt')
