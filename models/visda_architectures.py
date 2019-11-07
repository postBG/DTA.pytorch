import math

import torch.nn as nn
import torchvision.models.resnet as resnet

from models.resnet import Bottleneck, model_urls
from models.base import AbstractModel


def create_resnet_lower(model_name='resnet50', pretrained=True):
    """
    ResNet options: ResNet50, ResNet101, ResNet152
    :param model_name: resnet50, resnet101, resnet152
    :param pretrained: call pretrained model
    :return: resnet model
    """

    models = {'resnet50': resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152}

    return models[model_name](pretrained=pretrained, is_lower=True)


def create_resnet_upper(model_name='resnet50', pretrained=True, num_classes=12):
    """
    ResNet options: ResNet50, ResNet101, ResNet152
    :param model_name: resnet50, resnet101, resnet152
    :param pretrained: call pretrained model
    :return: resnet model
    """

    models = {'resnet50': resnet50,
              'resnet101': resnet101,
              'resnet152': resnet152}

    return models[model_name](pretrained=pretrained, is_lower=False, num_classes=num_classes)


def resnet50(pretrained=True, is_lower=True, num_classes=12, **kwargs):
    model = ResNetLower(Bottleneck, [3, 4, 6, 2], **kwargs) if is_lower else ResNetUpper(num_classes=num_classes)
    if pretrained:
        model.load_imagenet_state_dict(resnet.load_state_dict_from_url(model_urls['resnet50']))
        print("loaded imagenet pretrained resnet50")
    return model


def resnet101(pretrained=True, is_lower=True, num_classes=12, **kwargs):
    model = ResNetLower(Bottleneck, [3, 4, 23, 2], **kwargs) if is_lower else ResNetUpper(num_classes=num_classes)
    if pretrained:
        model.load_imagenet_state_dict(resnet.load_state_dict_from_url(model_urls['resnet101']))
        print("loaded imagenet pretrained resnet101")
    return model


def resnet152(pretrained=True, is_lower=True, num_classes=12, **kwargs):
    model = ResNetLower(Bottleneck, [3, 8, 36, 2], **kwargs) if is_lower else ResNetUpper(num_classes=num_classes)
    if pretrained:
        model.load_imagenet_state_dict(resnet.load_state_dict_from_url(model_urls['resnet152']))
        print("loaded imagenet pretrained resnet152")
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_dropout_after_first_bottleneck=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, use_dropout_after_first_bottleneck=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        dropout_module = nn.Dropout2d(0.1) if use_dropout_after_first_bottleneck else None
        layers.append(block(self.inplanes, planes, stride, downsample, dropout=dropout_module))
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

        h1 = self.layer4(x)
        h2 = self.layer4(x)

        return h1, h2

    def load_imagenet_state_dict(self, state_dict, strict=True):
        current_state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('layer4.2') or k.startswith('fc.'))}
        current_state_dict.update(state_dict)
        super().load_state_dict(current_state_dict, strict=strict)


class ResNetUpper(AbstractModel):
    def __init__(self, in_features=1000, num_classes=12):
        super().__init__()
        self.inplanes = 512 * 4
        self.layer4_last_conv = self._make_single_layer(Bottleneck, 512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512 * 4, 1000)
        self.fc2 = nn.Linear(1000, in_features)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features, num_classes)
        self.n_classes = num_classes
        self.drop_size = in_features
        # Added FC Layers

    def _make_single_layer(self, block, planes):
        return block(self.inplanes, planes)

    def load_imagenet_state_dict(self, state_dict, strict=True):
        current_state_dict = self.state_dict()
        state_dict = {k.replace('layer4.2', 'layer4_last_conv'): v for k, v in state_dict.items()
                      if k.startswith('layer4.2')}
        current_state_dict.update(state_dict)
        super().load_state_dict(current_state_dict, strict=strict)

    def forward(self, x, mask=None):
        x = self.layer4_last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        if mask is not None:
            x = mask * x
        x = self.fc3(x)
        return x
