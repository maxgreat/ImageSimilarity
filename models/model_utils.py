import torch
import torchvision
import torch.nn as nn

def copyParameters(net, modelBase):
    """
        Copy parameters from a model to another
        works with alexnet and vgg
    """
    #for each feature
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            #we copy convolution parameters
            f.weight.data = modelBase.features[i].weight.data
            f.bias.data = modelBase.features[i].bias.data

    #for each classifier element
    for i, f in enumerate(net.classifier):
        if type(f) is torch.nn.modules.linear.Linear:
            #we copy fully connected parameters
            if f.weight.size() == modelBase.classifier[i].weight.size():
                f.weight.data = modelBase.classifier[i].weight.data
                f.bias.data = modelBase.classifier[i].bias.data


def copyResNet(net, netBase):
    """
        TODO : make more general
    """
    net.conv1.weight.data = netBase.conv1.weight.data
    net.bn1.weight.data = netBase.bn1.weight.data
    net.bn1.bias.data = netBase.bn1.bias.data

    lLayer = [(net.layer1, netBase.layer1, 3),
              (net.layer2, netBase.layer2, 8),
              (net.layer3, netBase.layer3, 36),
              (net.layer4, netBase.layer4, 3)
             ]

    for targetLayer, rootLayer, nbC in lLayer:
        for i in range(nbC):
            targetLayer[i].conv1.weight.data = rootLayer[i].conv1.weight.data
            targetLayer[i].bn1.weight.data = rootLayer[i].bn1.weight.data
            targetLayer[i].bn1.bias.data = rootLayer[i].bn1.bias.data
            targetLayer[i].conv2.weight.data = rootLayer[i].conv2.weight.data
            targetLayer[i].bn2.weight.data = rootLayer[i].bn2.weight.data
            targetLayer[i].bn2.bias.data = rootLayer[i].bn2.bias.data
            targetLayer[i].conv3.weight.data = rootLayer[i].conv3.weight.data
            targetLayer[i].bn3.weight.data = rootLayer[i].bn3.weight.data
            targetLayer[i].bn3.bias.data = rootLayer[i].bn3.bias.data
        targetLayer[0].downsample[0].weight.data = rootLayer[0].downsample[0].weight.data
        targetLayer[0].downsample[1].weight.data = rootLayer[0].downsample[1].weight.data
        targetLayer[0].downsample[1].bias.data = rootLayer[0].downsample[1].bias.data
