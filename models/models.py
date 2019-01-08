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
        self.fc = None

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
    
    
    
class WeldonPooling(nn.Module):  #
    # Pytorch implementation of WELDON pooling

    def __init__(self, nMax=1, nMin=None):
        super(WeldonPooling, self).__init__()
        self.nMax = nMax
        if(nMin is None):
            self.nMin = nMax
        else:
            self.nMin = nMin

        self.input = torch.Tensor()
        self.output = torch.Tensor()
        self.indicesMax = torch.Tensor()
        self.indicesMin = torch.Tensor()

    def forward(self, input):

        self.batchSize = 0
        self.numChannels = 0
        self.h = 0
        self.w = 0

        if input.dim() == 4:
            self.batchSize = input.size(0)
            self.numChannels = input.size(1)
            self.h = input.size(2)
            self.w = input.size(3)
        elif input.dim() == 3:
            self.batchSize = 1
            self.numChannels = input.size(0)
            self.h = input.size(1)
            self.w = input.size(2)
        else:
            print('error in WeldonPooling:forward - incorrect input size')

        self.input = input

        nMax = self.nMax
        if nMax <= 0:
            nMax = 0
        elif nMax < 1:
            nMax = torch.clamp(torch.floor(nMax * self.h * self.w), min=1)

        nMin = self.nMin
        if nMin <= 0:
            nMin = 0
        elif nMin < 1:
            nMin = torch.clamp(torch.floor(nMin * self.h * self.w), min=1)

        x = input.view(self.batchSize, self.numChannels, self.h * self.w)

        # sort scores by decreasing order
        scoreSorted, indices = torch.topk(x, k=nMax, dim=x.dim() - 1)

        # compute top max
        self.indicesMax = indices[:, :, 0:nMax]
        self.output = torch.sum(scoreSorted[:, :, 0:nMax], dim=2, keepdim=True)
        self.output = self.output.div(nMax)

        # compute top min
        scoreSorted, indices = torch.topk(x*-1, k=nMin, dim=x.dim() - 1)
        scoreSorted = scoreSorted * -1

        if nMin > 0:
            self.indicesMin = indices[
                :, :, 0:nMin]
            yMin = torch.sum(
                scoreSorted[:, :, 0:nMin], 2, keepdim=True).div(nMin)
            self.output = torch.add(self.output, yMin)

        if input.dim() == 4:
            self.output = self.output.view(
                self.batchSize, self.numChannels, 1, 1)
        elif input.dim() == 3:
            self.output = self.output.view(self.numChannels, 1, 1)

        return self.output

    def backward(self, grad_output, _indices_grad=None):
        nMax = self.nMax
        if nMax <= 0:
            nMax = 0
        elif nMax < 1:
            nMax = torch.clamp(torch.floor(nMax * self.h * self.w), min=1)

        nMin = self.nMin
        if nMin <= 0:
            nMin = 0
        elif nMin < 1:
            nMin = torch.clamp(torch.floor(nMin * self.h * self.w), min=1)

        yMax = grad_output.clone().view(self.batchSize, self.numChannels,
                                        1).expand(self.batchSize, self.numChannels, nMax)
        z = torch.zeros(self.batchSize, self.numChannels,
                        self.h * self.w).type_as(self.input)
        z = z.scatter_(2, self.indicesMax, yMax).div(nMax)

        if nMin > 0:
            yMin = grad_output.clone().view(self.batchSize, self.numChannels, 1).div(
                nMin).expand(self.batchSize, self.numChannels, nMin)
            self.gradInput = z.scatter_(2, self.indicesMin, yMin).view(
                self.batchSize, self.numChannels, self.h, self.w)
        else:
            self.gradInput = z.view(
                self.batchSize, self.numChannels, self.h, self.w)

        if self.input.dim() == 3:
            self.gradInput = self.gradInput.view(
                self.numChannels, self.h, self.w)

        return self.gradInput



class ResNet_weldon(nn.Module):

    def __init__(self, pretrained=True, weldon_pretrained_path=None):
        super(ResNet_weldon, self).__init__()

        resnet = torchvision.models.resnet152(pretrained=pretrained)

        self.base_layer = nn.Sequential(*list(resnet.children())[:-2])
        self.spaConv = nn.Conv2d(2048, 2400, 1,)

        # add spatial aggregation layer
        self.wldPool = WeldonPooling(15)
        # Linear layer for imagenet classification
        self.fc = nn.Linear(2400, 1000)

        # Loading pretrained weights of resnet weldon on imagenet classification
        if pretrained:
            try:
                state_di = torch.load(
                    weldon_pretrained_path, map_location=lambda storage, loc: storage)['state_dict']
                self.load_state_dict(state_di)
            except Exception:
                print("Error when loading pretrained resnet weldon")

    def forward(self, x):
        x = self.base_layer(x)
        x = self.spaConv(x)
        x = self.wldPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
        



from sru import SRU


class SruEmb(nn.Module):
    def __init__(self, nb_layer, dim_in, dim_out, dropout=0.25):
        super(SruEmb, self).__init__()

        self.dim_out = dim_out
        self.rnn = SRU(dim_in, dim_out, num_layers=nb_layer,
                       dropout=dropout, rnn_dropout=dropout,
                       use_tanh=True, has_skip_term=True,
                       v1=True, rescale=False)

    def _select_last(self, x, lengths):
        batch_size = x.size(0)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        x = x.mul(mask)
        x = x.sum(1, keepdim=True).view(batch_size, self.dim_out)
        return x

    def _process_lengths(self, input):
        max_length = input.size(1)
        lengths = list(
            max_length - input.data.eq(0).sum(1, keepdim=True).squeeze())
        return lengths

    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        x = input.permute(1, 0, 2)
        x, hn = self.rnn(x)
        x = x.permute(1, 0, 2)
        if lengths:
            x = self._select_last(x, lengths)
        return x


class img_embedding(nn.Module):

    def __init__(self):
        super(img_embedding, self).__init__()
        model_weldon2 = ResNet_weldon(pretrained=True, weldon_pretrained_path="WELDON_CLASSIF_PRETRAINED")

        self.base_layer = nn.Sequential(*list(model_weldon2.children())[:-1])

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.base_layer(x)
        x = x.view(x.size()[0], -1)

        return x

    def get_activation_map(self, x):
        x = self.base_layer[0](x)
        act_map = self.base_layer[1](x)
        act = self.base_layer[2](act_map)
        return act, act_map


class joint_embedding(nn.Module):

    def __init__(self):
        super(joint_embedding, self).__init__()

        self.img_emb = torch.nn.DataParallel(img_embedding())
        self.cap_emb = SruEmb(4, 620, 2400)
        self.fc = torch.nn.DataParallel(nn.Linear(2400, 2400, bias=True))
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, imgs, caps, lengths):
        if imgs is not None:
            x_imgs = self.img_emb(imgs)
            x_imgs = self.dropout(x_imgs)
            x_imgs = self.fc(x_imgs)

            x_imgs = x_imgs / torch.norm(x_imgs, 2, dim=1, keepdim=True).expand_as(x_imgs)
        else:
            x_imgs = None

        if caps is not None:
            x_caps = self.cap_emb(caps, lengths=lengths)
            x_caps = x_caps / torch.norm(x_caps, 2, dim=1, keepdim=True).expand_as(x_caps)
        else:
            x_caps = None

        return x_imgs, x_caps






if __name__ == "__main__":
    print('Test model generation and forward')
    i = torch.rand(10,3,224,224).cuda()

    model = DoubleNet(resnetExtraction(),
                     resnetExtraction())

    o = imCap(i, i)
    print(o)
