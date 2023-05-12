import torch
import torch.nn.functional as F
from util.utils import set_seed, collate_fn, get_optimizer
from statistics import mean, median
'''
def get_centroids_for_virtual_centroids(args, model, dataset):
    checkpoint = torch.load(f"./models/{args.task_name}_loss_{args.loss}_best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    bank = None
    label_bank = None
    model.cuda()
    torch.cuda.manual_seed(1)
    for batch in dataset:
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels']
        if "deberta" in args.model_name_or_path:
            outputs = model.deberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        elif "roberta" in args.model_name_or_path:
            outputs = model.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        else:
            outputs = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        _, pooled = model.classifier(sequence_output)

        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach()
        else:
            bank = torch.cat([pooled.clone().detach()], dim=0)
            label_bank = torch.cat([labels.clone().detach(), label_bank], dim=0) 

    # Normalize the embeddings
    norm_bank = F.normalize(bank, dim=-1)
    N, d = norm_bank.size()
    print(f"Sample size: {N}, dim : {d}")

    return [norm_bank, label_bank] 
'''

def get_embeddings_for_virtual_centroids(args, model, dataset, norm = True, is_id = True):
    bank = None
    label_bank = None
    model.cuda()
    torch.cuda.manual_seed(1)
    for batch in dataset:
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels'] if is_id else None
        if "deberta" in args.model_name_or_path:
            outputs = model.deberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        elif "roberta" in args.model_name_or_path:
            outputs = model.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        else:
            outputs = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        _, pooled = model.classifier(sequence_output)
        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach() if is_id else None
        else:
            bank = torch.cat([pooled.clone().detach(), bank], dim=0)
            label_bank = torch.cat([labels.clone().detach(), label_bank], dim=0) if is_id else None
        
    # Normalize the embeddings
    norm_bank = F.normalize(bank, dim=-1)
    N, d = norm_bank.size()
    print(f"Sample size: {N}, dim : {d}")

    return [norm_bank, label_bank] if norm else [bank, label_bank]


def get_embeddings(args, model, dataset, norm=True, is_id = True):
    #checkpoint = torch.load(f"./models/{args.task_name}_loss_{args.loss}_best_model.pth")
    
    checkpoint = torch.load(f"./models/final_{args.task_name}_loss_{args.loss}_best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = get_optimizer(args, model)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    bank = None
    label_bank = None
    model.cuda()
    torch.cuda.manual_seed(1)
    for batch in dataset:
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        labels = batch['labels'] if is_id else None
        if "deberta" in args.model_name_or_path:
            outputs = model.deberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        elif "roberta" in args.model_name_or_path:
            outputs = model.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        else:
            outputs = model.bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            sequence_output = outputs[0]
        _, pooled = model.classifier(sequence_output)
        if bank is None:
            bank = pooled.clone().detach()
            label_bank = labels.clone().detach() if is_id else None
        else:
            bank = torch.cat([pooled.clone().detach(), bank], dim=0)
            label_bank = torch.cat([labels.clone().detach(), label_bank], dim=0) if is_id else None
        
    # Normalize the embeddings
    norm_bank = F.normalize(bank, dim=-1)
    N, d = norm_bank.size()
    print(f"Sample size: {N}, dim : {d}")

    return [norm_bank, label_bank] if norm else [bank, label_bank]


def get_optimal_virtual_centroids(args, model, dataset):
    pdist = torch.nn.PairwiseDistance(p=2)
    '''
    bank, label_bank = get_centroids_for_virtual_centroids(args, model, dataset)
    N, d = bank.size()

    #find each class centroids
    all_classes = list(set(label_bank.tolist()))

    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))
    '''
   
    bank, label_bank = get_embeddings_for_virtual_centroids(args, model, dataset)
    N, d = bank.size()
    #find each class centroids
    all_classes = list(set(label_bank.tolist()))

    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))

    class_mean = list(class_mean)
    for c in class_mean:
        c.requires_grad_()
    '''
    print('\nInitial distances : \n')
    for c in class_mean:
        print('\n')
        for c_ in class_mean:
            print(pdist(c, c_))
    '''
    print_freq = 50
    epochs = 500
    lr = 0.01
    #print('Loss : ')
    for epoch in range(epochs):
        loss = 0
        for c in class_mean:
            for c_ in class_mean:
                if(not torch.equal(c, c_)):
                    loss -= pdist(c, c_)
        
        loss.backward()
        with torch.no_grad():
             for c in class_mean:
                c -= lr * c.grad
                c /= torch.norm(c)
                
                c.grad = None
        '''
        if(epoch % print_freq == 0):
            print(loss)
        '''

    '''
    print('\nfinal distances')
    for c in class_mean:
        print('\n')
        for c_ in class_mean:
            print(pdist(c, c_))
    '''
    return class_mean


def analyze(args, model, dataset):

    bank, label_bank = get_embeddings(args, model, dataset)
    N, d = bank.size()
    #find each class centroids
    all_classes = list(set(label_bank.tolist())) 
    
    class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))
    
    # dot product with centroid pairs
    cluster = {}
    for c in all_classes:
        cluster[c] = []
        print(f"Dot Product {c} :")
        for opp in all_classes:
            dot_prod = torch.dot(bank[label_bank == opp].mean(0), class_mean[c])
            if opp != c:
                cluster[c].append(dot_prod.item())
                print(f"\t {opp} -> {dot_prod}")


    #Analysis
    print("=========================")
    print("Cluster Analysis....\n")
    print("=========================")
    for c in all_classes:
        print(f"Class #{c} : ")
        print(f"\tMean \t:: {mean(cluster[c])}")
        print(f"\tMedian \t:: {median(cluster[c])}")
        print(f"\tMax \t:: {max(cluster[c])}\t cluster_index \t:: {cluster[c].index(max(cluster[c]))}")
        print(f"\tMin \t:: {min(cluster[c])}\t cluster_index \t:: {cluster[c].index(min(cluster[c]))}")
