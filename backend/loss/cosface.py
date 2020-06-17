import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from backend.decorators import intent, loss

"""
code reference :
https://github.com/YirongMao/softmax_variants
"""

@intent
@loss
class CosFace(nn.Module):

    def __init__(self, label_dict,):
        super(CosFace, self).__init__()
        self.classes = len(label_dict)
        self.centers = nn.Parameter(torch.randn(self.classes, self.d_loss))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.cosface_m)
        margin_logits = self.cosface_s * (logits - y_onehot)
        return margin_logits

    def step(self, logits, feats, label, opts):
        mlogits = self(feats, label)

        total_loss = F.cross_entropy(mlogits, label)

        for opt in opts: opt.zero_grad()
        total_loss.backward()
        for opt in opts: opt.step()
        return total_loss
