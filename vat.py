import torch
import torch.nn as nn

from consistency_losses import KLDivLossWithLogits
from utils import disable_tracking_bn_stats


def l2_normalize(d):
    d_reshaped = d.view(d.size(0), -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VirtualAdversarialPerturbationGenerator(nn.Module):

    def __init__(self, feature_extractor, classifier, xi=1e-6, eps=3.5, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.kl_div = KLDivLossWithLogits()

    def forward(self, inputs):
        with disable_tracking_bn_stats(self.feature_extractor):
            with disable_tracking_bn_stats(self.classifier):
                features, _ = self.feature_extractor(inputs)
                logits = self.classifier(features).detach()

                # prepare random unit tensor
                d = l2_normalize(torch.randn_like(inputs).to(inputs.device))

                # calc adversarial direction
                for _ in range(self.ip):
                    x_hat = inputs + self.xi * d
                    x_hat.requires_grad = True
                    features_hat, _ = self.feature_extractor(x_hat)
                    logits_hat = self.classifier(features_hat)
                    adv_distance = self.kl_div(logits_hat, logits)
                    adv_distance.backward()
                    d = l2_normalize(x_hat.grad)
                    self.feature_extractor.zero_grad()

                r_adv = d * self.eps
                return r_adv.detach(), logits
