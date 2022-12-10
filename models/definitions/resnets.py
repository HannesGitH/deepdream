import os
from collections import namedtuple


import torch
from torchvision import models
from torch.hub import download_url_to_file


from utils.constants import *


class ResNet50(torch.nn.Module):

    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            resnet50 = models.resnet50(pretrained=True, progress=show_progress).eval()

        elif pretrained_weights in [SupportedPretrainedWeights.PLACES_365.name, SupportedPretrainedWeights.NSFW.name]:
            resnet50 = models.resnet50(pretrained=False, progress=show_progress).eval()

            binary_name = \
                'resnet50_places365.pth.tar' if pretrained_weights == SupportedPretrainedWeights.PLACES_365.name else\
                'ResNet50_nsfw_model.pth'
            binary_path = os.path.join(BINARIES_PATH, binary_name)

            if os.path.exists(binary_path):
                mmmm = torch.load(binary_path, map_location='cuda' if DEVICE.type == 'cuda' else 'cpu')
                state_dict = mmmm['state_dict'] if 'state_dict' in mmmm.keys() else mmmm
            else:
                binary_url = \
                    r'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar' if pretrained_weights == SupportedPretrainedWeights.PLACES_365.name else\
                    r'https://github.com/emiliantolo/pytorch_nsfw_model/raw/master/ResNet50_nsfw_model.pth'
                print(f'Downloading {binary_name} from {binary_url} it may take some time.')
                torch.hub.download_url_to_file(binary_url, binary_path)
                print('Done downloading.')
                state_dict = torch.load(binary_path)['state_dict']

            if pretrained_weights == SupportedPretrainedWeights.PLACES_365.name:
                new_state_dict = {}  # modify key names and make it compatible with current PyTorch model naming scheme
                for old_key in state_dict.keys():
                    new_key = old_key[7:]
                    new_state_dict[new_key] = state_dict[old_key]
            else:
                new_state_dict = state_dict

            if pretrained_weights == SupportedPretrainedWeights.PLACES_365.name:
                resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 365)
            elif pretrained_weights == SupportedPretrainedWeights.NSFW.name:
                resnet50.fc = torch.nn.Sequential(
                              torch.   nn.Linear(resnet50.fc.in_features, 512),
                              torch.   nn.ReLU(),
                              torch.   nn.Dropout(0.2),
                              torch.   nn.Linear(512, 10),
                              torch.   nn.LogSoftmax(dim=1))
            resnet50.load_state_dict(new_state_dict, strict=True)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model.')

        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'auxl1', 'auxl2', 'auxl3', 'auxl4', 'auxl5']

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # 3
        self.layer10 = resnet50.layer1[0]
        self.layer11 = resnet50.layer1[1]
        self.layer12 = resnet50.layer1[2]

        # 4
        self.layer20 = resnet50.layer2[0]
        self.layer21 = resnet50.layer2[1]
        self.layer22 = resnet50.layer2[2]
        self.layer23 = resnet50.layer2[3]

        # 6
        self.layer30 = resnet50.layer3[0]
        self.layer31 = resnet50.layer3[1]
        self.layer32 = resnet50.layer3[2]
        self.layer33 = resnet50.layer3[3]
        self.layer34 = resnet50.layer3[4]
        self.layer35 = resnet50.layer3[5]

        # 3
        self.layer40 = resnet50.layer4[0]
        self.layer41 = resnet50.layer4[1]
        # self.layer42 = resnet50.layer4[2]

        # Go even deeper into ResNet's BottleNeck module for layer 42
        self.layer42_conv1 = resnet50.layer4[2].conv1
        self.layer42_bn1 = resnet50.layer4[2].bn1
        self.layer42_conv2 = resnet50.layer4[2].conv2
        self.layer42_bn2 = resnet50.layer4[2].bn2
        self.layer42_conv3 = resnet50.layer4[2].conv3
        self.layer42_bn3 = resnet50.layer4[2].bn3
        self.layer42_relu = resnet50.layer4[2].relu

        self.avgpool = resnet50.avgpool

        if False:
            self.fc = resnet50.fc
        else:
            if pretrained_weights not in [SupportedPretrainedWeights.NSFW.name]:
                self.fc = resnet50.fc
                self.nswfnet = False
            elif pretrained_weights == SupportedPretrainedWeights.NSFW.name:
                self.nswfnet = True
                self.layer50 = resnet50.fc[0]
                self.layer51 = resnet50.fc[1]
                self.layer52 = resnet50.fc[2]
                self.layer53 = resnet50.fc[3]
                self.layer54 = resnet50.fc[4]

            # Set these to False so that PyTorch won't be including them in its autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        layer10 = x
        x = self.layer11(x)
        layer11 = x
        x = self.layer12(x)
        layer12 = x
        x = self.layer20(x)
        layer20 = x
        x = self.layer21(x)
        layer21 = x
        x = self.layer22(x)
        layer22 = x
        x = self.layer23(x)
        layer23 = x
        x = self.layer30(x)
        layer30 = x
        x = self.layer31(x)
        layer31 = x
        x = self.layer32(x)
        layer32 = x
        x = self.layer33(x)
        layer33 = x
        x = self.layer34(x)
        layer34 = x
        x = self.layer35(x)
        layer35 = x
        x = self.layer40(x)
        layer40 = x
        x = self.layer41(x)
        layer41 = x

        layer42_identity = layer41
        x = self.layer42_conv1(x)
        layer420 = x
        x = self.layer42_bn1(x)
        layer421 = x
        x = self.layer42_relu(x)
        layer422 = x
        x = self.layer42_conv2(x)
        layer423 = x
        x = self.layer42_bn2(x)
        layer424 = x
        x = self.layer42_relu(x)
        layer425 = x
        x = self.layer42_conv3(x)
        layer426 = x
        x = self.layer42_bn3(x)
        layer427 = x
        x += layer42_identity
        layer428 = x
        x = self.relu(x)
        layer429 = x

        x = self.avgpool(x)
        layer43 = x

        x = x.reshape(x.shape[0], -1)

        if False:
            pass
        else:
            if self.nswfnet:
                x = self.layer50(x)
                layer50 = x
                x = self.layer51(x)
                layer51 = x
                # print(layer51[0].argmax())
                # print(layer51[0])
                # x = self.layer52(x)
                # layer52 = x
                x = self.layer53(x)
                layer53 = x
                # x = self.layer54(x)
                x = x+torch.ones_like(x)
                x = torch.nn.Sigmoid()(x)[:,:5]
                layer54 = x
                # print(layer54)
            else:
                x = self.fc(x)
                layer5 = x

        # Feel free to experiment with different layers, layer35 is my favourite
        net_outputs = namedtuple("ResNet50Outputs", self.layer_names)
        # You can see the potential ambiguity arising here if we later want to reconstruct images purely from the filename
        if not self.nswfnet:
            out = net_outputs(layer10, layer23, layer34, layer429, layer5, layer43, layer35, layer32, layer30, layer22)
        else:
            out = net_outputs(layer50, layer51, layer429, layer53, layer54, layer43, layer35, layer32, layer30, layer22)
        
        return out