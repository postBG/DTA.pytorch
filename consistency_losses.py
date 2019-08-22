from torch import nn as nn
from torch.nn import functional as F


class AbstractConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits1, logits2):
        raise NotImplementedError


class LossWithLogits(AbstractConsistencyLoss):
    def __init__(self, reduction='mean', loss_cls=nn.L1Loss):
        super().__init__(reduction)
        self.loss_with_softmax = loss_cls(reduction=reduction)

    def forward(self, logits1, logits2):
        loss = self.loss_with_softmax(F.softmax(logits1, dim=1), F.softmax(logits2, dim=1))
        return loss


class DiscrepancyLossWithLogits(AbstractConsistencyLoss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = LossWithLogits(reduction=reduction, loss_cls=nn.L1Loss)

    def forward(self, logits1, logits2):
        return self.loss(logits1, logits2)


class KLDivLossWithLogits(AbstractConsistencyLoss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, logits1, logits2):
        return self.kl_div_loss(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1))


def get_consistency_loss(loss_name, reduction='mean'):
    if loss_name == 'l1':
        loss = LossWithLogits(loss_cls=nn.L1Loss, reduction=reduction)
    elif loss_name == 'l2':
        loss = LossWithLogits(loss_cls=nn.MSELoss, reduction=reduction)
    elif loss_name == 'kld':
        loss = KLDivLossWithLogits(reduction=reduction)
    else:
        raise ValueError("There is not matched loss with {}".format(loss_name))

    return loss
