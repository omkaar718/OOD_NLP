from datasets import load_dataset
from data.loader import *
import random

dataset_map = {
    "sst2" : {
        "keys" : ["sentence", None],
        "datasets" : load_sst2,
    },
    "rte" : {
        "keys" : ["sentence1", "sentence2"],
        "datasets" : load_glue,
    },
    "20ng" : {
        "keys" : ["text", None],
        "datasets" : load_20ng,
    },
    "imdb" : {
        "keys" : ["text", None],
        "datasets" : load_imdb,
    },
    "multi30k" : {
        "keys" : ["text", None],
        "datasets" : load_multi30k,
    },
    "trec" : {
        "keys" : ["text", None],
        "datasets" : load_trec,
    },
    "wmt16" : {
        "keys" : ["en", None],
        "datasets" : load_wmt16,
    },
    "mnli" : {
        "keys" : ["premise", "hypothesis"],
        "datasets" : load_glue_mnli,
    }
}

def preprocess_function(examples, keys, tokenizer, max_seq_length=256, is_id=False):
    inputs = (
        (examples[keys[0]],) if keys[1] is None else (examples[keys[0]] + " " + examples[keys[1]],)
    )
    result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
    result["labels"] = examples["label"] if 'label' in examples else 0
    return result

def load_dataset(task_name, tokenizer, max_seq_length=256, is_id=False):

    print("Loading {}".format(task_name))
    keys = dataset_map[task_name]["keys"]
    datasets = dataset_map[task_name]["datasets"]()

    train_dataset = dev_dataset = test_dataset = None

    if 'train' in datasets and is_id:
        train_dataset = [preprocess_function(i, keys, tokenizer, max_seq_length) for i in datasets['train']]
    
    if 'validation' in datasets and is_id:    
        dev_dataset = [preprocess_function(i, keys, tokenizer, max_seq_length) for i in datasets['validation']]

    if 'test' in datasets:
        test_dataset = [preprocess_function(i, keys, tokenizer, max_seq_length) for i in datasets['test']]

    return train_dataset, dev_dataset, test_dataset