# CS769_Project

# Requirements

```
pip3 install -r requirements.txt
```

# Dataset

Datasets used in the paper is automatically downloaded by the dataset package. Required datasets are placed with the data/. folder

# Training

```
python3 main_final.py --model_name_or_path roberta-large --loss multi_task --task_name trec --centroids true --project_name final_trec --learning_rate 1e-6 --batch_size 16 --viz false
```
The `task_name` parameter can take `sst2`, `trec`, or `20ng`

Note: Keep --viz false if you want to avoid wandb visualizations and instead print on terminal. If cuda out of memory error occurs, please reduce the batch size using --batch_size argument in the command.


# Analysis

Analysis script under util/analysis.py. This would help in analyzing the cluster centroids and their distances.
```
python3 main_final.py --model_name_or_path roberta-large --loss multi_task --task_name trec --viz false --analysis true
```
If cuda out of memory error occurs, please reduce the batch size using --batch_size argument in the command.

# Plots

Plotting script under util/plot.py. This would visualize the embeddings of ID and OOD data using t-SNE.
```
python3 main_final.py --model_name_or_path roberta-large --loss multi_task --task_name trec --viz false --plot true  
```
If cuda out of memory error occurs, please reduce the batch size using --batch_size argument in the command.

# Authors

- Omkar Chandrakant Prabhune (oprabhune@wisc.edu)
- Sourav Suresh (sourav.suresh@wisc.edu)
