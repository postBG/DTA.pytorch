import torch
import torch.nn as nn
import torch.nn.functional as F

from consistency_losses import KLDivLossWithLogits
from vat import VirtualAdversarialPerturbationGenerator


class GANLoss(nn.Module):
    def __init__(self, source_label=0.0, target_label=1.0, device='cuda'):
        super(GANLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.source_label = source_label
        self.target_label = target_label
        self.device = device

    def __call__(self, inputs, to_source):
        """
        :param inputs: logit, output from discriminator
        :param to_source: True if fool as source, False if fool as target
        :return: GAN Loss
        """
        if to_source:
            target_tensor = torch.LongTensor(inputs.size(0)).fill_(self.source_label).to(self.device)
        else:
            target_tensor = torch.LongTensor(inputs.size(0)).fill_(self.target_label).to(self.device)

        return self.loss(inputs, target_tensor)


class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        elementwise_entropy = -p * F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if self.reduction == 'sum':
            return sum_entropy

        return torch.mean(sum_entropy)


class ClassBalanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        cls_balance = -torch.mean(torch.log(torch.mean(p, 0) + 1e-6))
        return cls_balance


class LDSLoss(nn.Module):
    def __init__(self, model, xi=1e-6, eps=2.0, ip=1):
        super().__init__()
        self.model = model
        self.vap_generator = VirtualAdversarialPerturbationGenerator(model, xi=xi, eps=eps, ip=ip)
        self.kl_div = KLDivLossWithLogits()

    def forward(self, inputs):
        r_adv, logits = self.vap_generator(inputs)

        adv_inputs = inputs + r_adv
        adv_logits = self.model(adv_inputs)
        lds_loss = self.kl_div(adv_logits, logits)

        return lds_loss
