import torch
from copy import deepcopy


def create_adversarial_dropout_mask(mask, jacobian, delta):
    """

    :param mask: shape [batch_size, ...]
    :param jacobian: shape [batch_size, ...]
    :param delta:
    :return:
    """
    num_of_units = int(torch.prod(torch.tensor(mask.size()[1:])).to(torch.float))
    change_limit = int(num_of_units * delta)
    mask = (mask > 0).to(torch.float)

    if change_limit == 0:
        return deepcopy(mask).detach(), torch.Tensor([]).type(torch.int64)

    # mask (mask=1 -> m = 1), (mask=0 -> m=-1)
    m = 2 * mask - torch.ones_like(mask)

    # sign of Jacobian  (J>0 -> s=1), (J<0 -> s=-1)
    s = torch.sign(jacobian)

    # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
    change_candidates = ((m * s) < 0).to(torch.float)

    # ordering abs_jacobian for candidates
    # the maximum number of the changes is "change_limit"
    # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" )
    abs_jacobian = torch.abs(jacobian)
    candidate_abs_jacobian = (change_candidates * abs_jacobian).view(-1, num_of_units)
    topk_values, topk_indices = torch.topk(candidate_abs_jacobian, change_limit + 1)
    min_values = topk_values[:, -1]
    change_target_marker = (candidate_abs_jacobian > min_values.unsqueeze(-1)).view(mask.size()).to(torch.float)

    # changed mask with change_target_marker
    adv_mask = torch.abs(mask - change_target_marker)

    # normalization
    adv_mask = adv_mask.view(-1, num_of_units)
    num_of_undropped_units = torch.sum(adv_mask, dim=1).unsqueeze(-1)
    adv_mask = ((adv_mask / num_of_undropped_units) * num_of_units).view(mask.size())

    return adv_mask.clone().detach(), (adv_mask == 0).nonzero()[:, 1]


def calculate_jacobians(h, clean_logits, classifier, fc_mask_size, consistency_criterion, reset_grad_fn):
    cnn_mask = torch.ones((*h.size()[:2], 1, 1)).to(h.device)
    fc_mask = torch.ones(cnn_mask.size(0), fc_mask_size).to(cnn_mask.device)
    cnn_mask.requires_grad = True
    fc_mask.requires_grad = True

    h_logits = classifier(cnn_mask * h, fc_mask)
    discrepancy = consistency_criterion(h_logits, clean_logits)
    discrepancy.backward()

    reset_grad_fn()
    return cnn_mask.grad.clone(), fc_mask.grad.clone(), h_logits
