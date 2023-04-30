import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from util.analysis import get_embeddings


class CompLoss(nn.Module):
    '''
    Compactness Loss with class-conditional prototypes
    '''
    def __init__(self, args, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        prototypes = F.normalize(prototypes, dim=1) 
        proxy_labels = torch.arange(0, self.args.n_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda() #bz, cls

        # compute logits
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss

class DisLoss(nn.Module):
    '''
    Dispersion Loss with EMA prototypes
    '''
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
        super(DisLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        # self.register_buffer("prototypes", torch.zeros(self.args.n_cls,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.prototypes = self.init_class_prototypes()

    def forward(self, features, labels):    

        prototypes = self.prototypes
        num_cls = self.args.n_cls
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()


        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        """
            Initialize class prototypes
        """
        start = time.time()

        bank, label_bank = get_embeddings(self.args, self.model, self.loader,load=False)
        N, d = bank.size()

        #find each class centroids
        all_classes = list(set(label_bank.tolist())) 
        class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
        for c in all_classes:
            class_mean[c] = (bank[label_bank == c].mean(0))

        # measure elapsed time
        duration = time.time() - start

        # @TODO: CHeck
        print(f'Time to initialize class mean: {duration:.3f}')
        class_mean = F.normalize(class_mean, dim=1)

        print("Final prototype size :: ", class_mean.size())
        return class_mean

