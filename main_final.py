import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer
from transformers import BertConfig, BertTokenizer
from transformers import DebertaConfig, DebertaTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from util.utils import set_seed, collate_fn, get_optimizer
from util.analysis import analyze, get_optimal_virtual_centroids
from util.plot import plot
from datasets import load_metric
from models.deberta import DebertaForSequenceClassification
from models.roberta import RobertaForSequenceClassification
from models.bert import BertForSequenceClassification
from util.evaluation import evaluate_ood
import wandb
import warnings
from data.dataset import load_dataset


warnings.filterwarnings("ignore")


task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
}


task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
}

def train(args,num_labels, optimal_centroids,  model, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    if(args.task_name in ['20ng', 'imdb']):
        batch_size = int(args.batch_size/2)
    else:
        batch_size = args.batch_size
            
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = get_optimizer(args, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def detect_ood():
        model.prepare_ood(dev_dataloader)
        for tag, ood_features in benchmarks:
            results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag)
            wandb.log(results, step=num_steps) if args.viz.lower() == "true" else print("\t OOD Eval Results :: ", results)

    num_steps = 0
    
    
    val_acc = 0
    for epoch in range(int(args.num_train_epochs)):
        print('\nEpoch : ', epoch) 
        model.zero_grad()
        loss = cos_loss = size = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            # print('\nOptimal centroids : ',  optimal_centroids)
            model.train()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(num_labels, optimal_centroids, **batch)
            
            loss, cos_loss = outputs[0], outputs[1]
            # print(loss, cos_loss)
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if args.viz.lower() == "true":
                wandb.log({'loss': loss.item()}, step=num_steps)
                wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)
            else:
                loss += float(loss.item())
                cos_loss += float(cos_loss.item())
                size += 1
        if args.viz.lower() == "false": 
            print(f"Epoch #{epoch} results:: ")
            print(f"\tLoss :: {loss/size}")
            print(f"\tCos_Loss :: {cos_loss/size}")
        

        results = evaluate(args, num_steps, num_labels, optimal_centroids, model, dev_dataset, tag="dev")
        if results['dev_accuracy'] > val_acc:
            val_acc = results['dev_accuracy']
            print(f"\t [Checkpoint] Found Better Dev Accuracy :: {val_acc}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': {loss/size},
                'val_acc': {val_acc},
                'cos_loss': {cos_loss/size},
                }, f"./models/final_{args.task_name}_loss_{args.loss}_best_model.pth")
        
        wandb.log(results, step=num_steps) if args.viz.lower() == "true" else print("\t Dev Validation :: ", results)
        #results = evaluate(args, num_steps, num_labels, optimal_centroids, model, test_dataset, tag="test")
        #wandb.log(results, step=num_steps) if args.viz.lower() == "true" else print("\t Test Results :: ", results)
        

    checkpoint = torch.load(f"./models/final_{args.task_name}_loss_{args.loss}_best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_acc = checkpoint['val_acc']
    cos_loss = checkpoint['cos_loss']
    print(f"\t Evaluating best model got at epoch # :{epoch}")
    results = evaluate(args, num_steps, num_labels, optimal_centroids, model, test_dataset, tag="test")
    wandb.log(results, step=num_steps) if args.viz.lower() == "true" else print("\t Test Results :: ", results)

    detect_ood()
    


def evaluate(args, num_steps, num_labels, optimal_centroids, model, eval_dataset, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result
    
    if(args.task_name in ['20ng', 'imdb']):
        batch_size = int(args.batch_size/2)
    else:
        batch_size = args.batch_size

    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

    label_list, logit_list = [], []
    total_loss = total_cos_loss = 0
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        # batch["labels"] = None
        outputs = model(num_labels, optimal_centroids,**batch)
        loss, cos_loss = outputs[0], outputs[1]
        total_loss += loss.item()
        total_cos_loss += cos_loss.item()
        logits = outputs[2].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)

    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    
    total_loss /= step
    total_cos_loss /= step
    print('\n\n')
    #print(f"Loss for tag {tag} at num_steps {num_steps} = {total_loss}")
    #print(f"cos_loss for tag {tag} at num_steps {num_steps} = {total_cos_loss}")
    if(args.viz.lower() == 'true'):
        wandb.log({f'{tag}_loss': total_loss}, step=num_steps)
        wandb.log({f'{tag}_cos_loss': total_cos_loss}, step=num_steps)

    return results


def main():
    parser = argparse.ArgumentParser()
    # Options :: roberta-large
    parser.add_argument("--model_name_or_path", default="microsoft/deberta-base", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=15, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    parser.add_argument("--viz", type=str, default="true")
    parser.add_argument("--analysis", type=str, default="false")
    parser.add_argument("--plot", type=str, default="false")
    parser.add_argument("--centroids", type=str, default="false")
    args = parser.parse_args()

    if args.viz.lower() == "true":
        wandb.init(project=args.project_name, name=args.task_name + '-' + str(args.alpha) + "_" + args.loss)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    num_labels = task_to_labels[args.task_name]

    if "deberta" in args.model_name_or_path:
        print("Training with Deberta")
        config = DebertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = DebertaTokenizer.from_pretrained(args.model_name_or_path)
        model = DebertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)
    elif "roberta" in args.model_name_or_path:
        print("Training with Roberta")
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)
        
    elif args.model_name_or_path.startswith('bert'):
        print("Training with BERT Model")
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)
    else:
        raise Exception("Currently only BERT, Roberta, Deberta Models are supported")

    datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    #datasets = ['trec', 'multi30k']
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load_dataset(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True)
        
        else:
            _, _, ood_dataset = load_dataset(dataset, tokenizer, max_seq_length=args.max_seq_length)
            benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
        
    
    if args.plot.lower() == "true":
        in_data = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        ood_dataset = []
        for _ in benchmarks:
            ood_dataset += _[1]
        ood_data = DataLoader(ood_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        plot(args, model, in_data, ood_data, args.task_name, num_labels)
        
    if args.analysis.lower() == "true":
        data = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
        analyze(args, model, data)

    #if args.plot.lower() == "false" and args.analysis.lower() == "false":      
    #    train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)

    if args.centroids.lower() == "true":
        data =  DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
        print("Computing optimal centroids (step 1 in Section 3.3 of the project report)...")
        optimal_centroids = get_optimal_virtual_centroids(args, model, data)
        print('Optimal centroids obtained')
        #if args.plot.lower() == "false" and args.analysis.lower() == "false":
    
        train(args, num_labels, optimal_centroids, model, train_dataset, dev_dataset, test_dataset, benchmarks)
if __name__ == "__main__":
    main()
