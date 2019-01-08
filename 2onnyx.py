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
    dummy_input = torch.randn(20, 3, 224, 224)
    print('Output size:', model(dummy_input).shape)
    output_names = [ "output"]
    torch.onnx.export(model, dummy_input, saveName, verbose=True, output_names=output_names)
    
    
def text2onnx(model, saveName):
    dummy_input = torch.randn(5,10,620)
    print('Output text:', model(dummy_input).shape)
    output_names = [ "output"]
    torch.onnx.export(model, dummy_input, saveName, verbose=True, output_names=output_names)
    
    
def loadFullNet(modelPath):
    je = md.joint_embedding()
    a = torch.load(modelPath)
    je.load_state_dict(a['state_dict'])
    imageEmbed = Sequential(*list(je.img_emb.module.children()))
    rnn = je.cap_emb
    return imageEmbed, rnn



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_name", default='data/pretrained_classif_152_2400.pth.tar')
    parser.add_argument("-s", "--save_name", default='resnet.onnx')
    parser.add_argument("-t", "--model_type", help="Should be weldon or full", default="weldon")


    args = parser.parse_args()
    
    if args.model_type == "weldon":
        model = weldon2resnet(args.model_name)
    else:
        model, textmodel = loadFullNet(args.model_name)
        #text2onnx(textmodel, "textEmbed.onnx")
    toonnx(model, args.save_name)
    

    

