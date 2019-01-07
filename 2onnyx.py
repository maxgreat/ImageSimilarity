import torch
from torch.nn import DataParallel, Sequential
import argparse
import h5py

import torch.onnx
import torchvision
#import uff
import numpy as np

import models.models as md


def weldon2resnet(name):
    model = md.ResNet_weldon(weldon_pretrained_path=name)
    model = model.base_layer
    model.add_module('8', torch.nn.AvgPool2d( (7,7), (1,1) ) )
    modules = list(model.children())
    model = Sequential(*modules)
    return model
    
def toonnx(model, saveName):
    dummy_input = torch.randn(1000, 3, 224, 224)
    #print('Output size:', model(dummy_input).shape)
    output_names = [ "output"]
    torch.onnx.export(model, dummy_input, saveName, verbose=True, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='data/pretrained_classif_152_2400.pth.tar')
    parser.add_argument("--save_name", default='resnet.onnx')


    args = parser.parse_args()
    
    model = weldon2resnet(args.model_name)
    toonnx(model, args.save_name)
    
    

    

