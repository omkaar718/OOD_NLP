import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, RobertaModel
from sklearn.covariance import EmpiricalCovariance

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = pooled = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, pooled


class RobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        num_labels,
        optimal_centroids,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        pdist =  pdist = torch.nn.PairwiseDistance(p=2)
        # print('\nLabels received : ', type(labels), labels.size(), labels)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)
        #bank = None
        #label_bank = None

        loss = None
        #cos_loss = torch.Tensor(0).cuda()
        if labels is not None:
            if self.config.loss == 'margin':
                dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                max_dist = (dist * mask).max()
                cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()
            
            elif(self.config.loss == 'custom-2'):
                norm_pooled = F.normalize(pooled, dim=-1)
                dist = ((norm_pooled.unsqueeze(1) - norm_pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) - (dist  * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()        

            elif(self.config.loss == 'stage_2_compactness'):
                norm_pooled = F.normalize(pooled, dim=-1)
                dist = ((norm_pooled.unsqueeze(1) - norm_pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                # neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3)
                # - (dist  * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()

            elif(self.config.loss in ['custom-3', 'stage2_centroids'] ):
                '''
                if bank is None:
                    bank = pooled
                    label_bank = labels.clone().detach() if is_id else None
                else:
                    print('\nIn bank not None')
                    bank = torch.cat([pooled.clone().detach(), bank], dim=0)
                    label_bank = torch.cat([labels.clone().detach(), label_bank], dim=0) if is_id else None
                '''

                norm_pooled = F.normalize(pooled, dim=-1)
                N, d = norm_pooled.size()
                class_mean = torch.zeros(num_labels, d).cuda()
                
                for c in range(num_labels):
                    this_class_elements =  norm_pooled[labels.clone().detach() == c]
                    if(this_class_elements.numel()):
                        class_mean[c] = this_class_elements.mean(0)
                    #class_mean[c] = norm_pooled[labels.clone().detach() == c].mean(0)


                #print('\nCurrent Class mean : ',class_mean)
                cos_loss = 0
                for n in range(num_labels):
                    
                    if(not torch.equal(class_mean[n], torch.zeros(d).cuda())):
                        #print('Contributing to loss : ', n)
                        distance_ = pdist(class_mean[n], optimal_centroids[n])
                        # print('\ndistance : ', distance_)
                        cos_loss += distance_
                        # print('cos_loss at n  ',n,  cos_loss)
                #cos_loss = cos_loss.mean()
                #print('Mean cos_loss : ', cos_loss)

            elif(self.config.loss == 'multi_task'):
                norm_pooled = F.normalize(pooled, dim=-1)
                dist = ((norm_pooled.unsqueeze(1) - norm_pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                cos_loss_1 = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) - (dist  * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss_1 = cos_loss_1.mean()

                # loss 2
                norm_pooled = F.normalize(pooled, dim=-1)
                N, d = norm_pooled.size()
                class_mean = torch.zeros(num_labels, d).cuda()


                for c in range(num_labels):
                    this_class_elements =  norm_pooled[labels.clone().detach() == c]
                    if(this_class_elements.numel()):
                        class_mean[c] = this_class_elements.mean(0)
                    #class_mean[c] = norm_pooled[labels.clone().detach() == c].mean(0)


                #print('\nCurrent Class mean : ',class_mean)
                cos_loss_2 = 0
                for n in range(num_labels):

                    if(not torch.equal(class_mean[n], torch.zeros(d).cuda())):
                        #print('Contributing to loss : ', n)
                        distance_ = pdist(class_mean[n], optimal_centroids[n])
                        # print('\ndistance : ', distance_)
                        cos_loss_2 += distance_
                cos_loss_2 /= num_labels
                cos_loss = cos_loss_1 + cos_loss_2


            elif(self.config.loss == 'supcon'):
                
                norm_pooled = F.normalize(pooled, dim=-1)
                cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
                mask = mask - torch.diag(torch.diag(mask))
                cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
                cos_loss = -torch.log(cos_loss + 1e-5)
                cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()

            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss = loss + self.config.alpha * cos_loss
        output = (logits,) + outputs[2:]
        output = output + (pooled,)
        #print('\nLoss, cos_loss', loss, cos_loss)
        return ((loss, cos_loss) + output) if loss is not None else output

    def compute_ood(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)

        ood_keys = None
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

        maha_score = []
        for c in self.all_classes:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score = maha_score.min(-1)[0]
        maha_score = -maha_score

        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]

        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
        }
        return ood_keys

    def prepare_ood(self, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
            logits, pooled = self.classifier(sequence_output)
            if self.bank is None:
                self.bank = pooled.clone().detach()
                self.label_bank = labels.clone().detach()
            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                self.bank = torch.cat([bank, self.bank], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)

        self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.all_classes = list(set(self.label_bank.tolist()))
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()
        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()
