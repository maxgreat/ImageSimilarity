import torch
import torchvision.models
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as f

import fastText

from models import model_utils
#import model_utils

class AlexNetExtraction(nn.Module):

    def __init__(self, features_size=4096):
        super(AlexNetExtraction, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, features_size)
        )
        model_utils.copyParameters(self, torchvision.models.alexnet(pretrained=True))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class ResnetExtraction(torchvision.models.ResNet):

    def __init__(self, block, layers, size=2048):
        #super(ResnetExtraction, self).__init__(block, layers)
        super().__init__(block, layers)
        self.size=size
        self.fc = nn.Linear(2048, size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if not self.size == 2048:
            x = self.fc(x)

        return x

def resnetExtraction(pretrained=True, size=2048):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResnetExtraction(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], size=size)
    if pretrained:
        model_utils.copyResNet(model, torchvision.models.resnet152(pretrained=True))
        #model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth'))

    return model



class ImageCaptionNet(nn.Module):
    def __init__(self, ImageNetwork):
        super(ImageCaptionNet, self).__init__()
        self.ImageNet = nn.DataParallel(ImageNetwork)
        #self.TextNet = TextNetwork
        #self.TextNet = nn.DataParallel(nn.GRU(300, 2048))
        self.TextNet = nn.GRU(300, 2048, num_layers=4, dropout=0.1, batch_first=False).cuda()

    def forward(self, i, t):
        i = self.ImageNet(i)
        f.normalize(i,p=2, dim=-1)
        t = self.TextNet(t, torch.zeros(4,t.shape[1], 2048).cuda())[-1]
        f.normalize(t,p=2, dim=-1)
        return i,t

class SiameseNet(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.net = nn.DataParallel(network)

    def forward(self, x1, x2):
        return self.net(x1), self.net(x2)

class DoubleNet(nn.Module):
    def __init__(self, net1, net2, norm=True):
        super().__init__()
        self.net1 = nn.DataParallel(net1)
        #self.net1 = net1
        #torch.distributed.init_process_group(backend="nccl")
        #self.net2 = nn.parallel.distributed.DistributedDataParallel(net2)
        self.net2 = net2
        self.norm = norm

    def forward(self, x1, x2):
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        if self.norm :
            f.normalize(x1,p=2, dim=-1)
            f.normalize(x2,p=2, dim=-1)
        return x1, x2


class TripletNet(nn.Module):
    def __init__(self, net1, net2, norm=True):
        super().__init__()
        self.net1 = nn.DataParallel(net1)
        self.net2 = net2
        self.norm = norm

    def forward(self, x1, x2, x3):
        x1 = self.net1(x1)
        x2 = self.net2(x2)
        x3 = self.net2(x3)
        if self.norm :
            f.normalize(x1,p=2, dim=-1)
            f.normalize(x2,p=2, dim=-1)
            f.normalize(x3,p=2, dim=-1)
        return x1, x2, x3

    def LoadFromDouble(self, doubleNet):
        self.net1 = doubleNet.net1
        self.net2 = doubleNet.net2

def ResnetEmbedding():
    embed = nn.EmbeddingBag(99999,300)
    #embed = torch.load('/workspace/imageSimilarity/embedding/embedding.save')
    net = DoubleNet(resnetExtraction(size=300), embed)
    return net

if __name__ == "__main__":
    print('Test model generation and forward')
    i = torch.rand(10,3,224,224).cuda()

    model = DoubleNet(resnetExtraction(),
                     resnetExtraction())

    o = imCap(i, i)
    print(o)
