import math

import torch.nn as nn
import torchvision.models.resnet as resnet

from models.resnet import Bottleneck, model_urls
from models.base import AbstractModel


def create_resnet_model(model_name='resnet50', pretrained=True):
    """
    ResNet options: ResNet50, ResNet101, ResNet152
    :param model_name: resnet50, resnet101, resnet152
    :param pretrained: call pretrained model
    :return: resnet model
    """

    models = {'resnet50': resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152}

    return models[model_name](pretrained=pretrained)


def resnet50(pretrained=True, **kwargs):
    model = ResNetLower(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(resnet.model_zoo.load_url(model_urls['resnet50']))
        print("loaded pretrained resnet50")
    return model


def resnet101(pretrained=True, **kwargs):
    model = ResNetLower(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(resnet.model_zoo.load_url(model_urls['resnet101']))
        print("loaded pretrained resnet101")
    return model


def resnet152(pretrained=True, **kwargs):
    model = ResNetLower(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(resnet.model_zoo.load_url(model_urls['resnet152']))
        print("loaded pretrained resnet152")
    return model


class ResNetLower(AbstractModel):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetLower, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, 1000)
        # self.num_out_features = 512 * block.expansion

        # Added FC Layers
        self.fc1 = nn.Linear(512 * block.expansion, 1000)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def set_bn_momentum(self, momentum):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum

    def freeze_bn(self, freeze_bn):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() if freeze_bn else m.train()

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
        x = self.fc1(x)
        x = self.relu(x)

        x1 = self.dropout(x)
        x2 = self.dropout(x)

        h1 = self.relu(self.fc2(x1))
        h2 = self.relu(self.fc2(x2))
        return h1, h2

    def load_state_dict(self, state_dict, strict=True):
        current_state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        current_state_dict.update(state_dict)
        super().load_state_dict(current_state_dict, strict=strict)


class ResNetUpper(AbstractModel):
    def __init__(self, in_features=1000, num_classes=12):
        super().__init__()
        self.fc3 = nn.Linear(in_features, num_classes)
        self.n_classes = num_classes

    def forward(self, x):
        x = self.fc3(x)
        return x
