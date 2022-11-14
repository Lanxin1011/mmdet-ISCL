from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes). I think the shape should be (batch_size).
        """
        n = inputs.size(0)  # batch_size

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())  # 1 for the same class, 0 for different classes
        dist_ap, dist_an = [], []
        # print(f'mask:\n{mask}')
        # print(f'dist:\n{dist}')

        valid = (mask[0] == 1).any() and (mask[0] == 0).any()  # Determine whether there are samples of the same category
                                                               # and samples of different categories, or we can not calculate the triplet loss
        if valid:
            for i in range(n):  # traverse every anchor feature
                if valid:
                    dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                    dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)

            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            print(f'mask:\n{mask}')
            print(f'dist:\n{dist}')
            loss = 0
        return loss