import torch
from torch.nn import DataParallel, Sequential
import argparse
import h5py

import torch.onnx
import torchvision
#import uff
import numpy as np



def weldon2resnet():
    model = misc.model.ResNet_weldon('aa',weldon_pretrained_path='data/pretrained_classif_152_2400.pth.tar')
    model = model.base_layer
    model.add_module('8', torch.nn.AvgPool2d( (7,7), (1,1) ) )
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--save_name", default='model.onnx')
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", type=int, default=1000)


    args = parser.parse_args()

    model= torch.load(args.model_name)
    #model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = model.module
    modules = list(model.children())[:-1]
    model = Sequential(*modules)
    dummy_input = torch.randn(args.batch_size, 3, 224, 224)
    print(model(dummy_input).shape)
    output_names = [ "output"]
    torch.onnx.export(model, dummy_input, args.save_name, verbose=True, output_names=output_names)

