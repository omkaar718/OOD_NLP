# CS769_Project

# Requirements

```
pip3 install -r requirements.txt
```

# Dataset

Datasets used in the paper is automatically downloaded by the dataset package. Required datasets are placed with the data/. folder

# Training

```
python3 main_with_vis_and_multi_gpu.py --model_name_or_path roberta-large --loss custom --task_name 20ng --viz true --project_name hw3_20ng_custom-2-lr_1e-5 --learning_rate 1e-5
```
The `task_name` parameter can take `sst2`, `imdb`, `trec`, or `20ng`

Note: Keep --viz false if you want to avoid wandb viz and instead print on terminal. If cuda out of memory error occurs, please reduce the batch size using --batch_size argument in the command.


# Analysis

Analysis script under util/analysis.py. This would help in analyzing the cluster centroids and their distances.
```
python3 main_with_vis_and_multi_gpu.py --model_name_or_path roberta-large --loss custom --task_name 20ng --analysis true
```
If cuda out of memory error occurs, please reduce the batch size using --batch_size argument in the command.

# Plots

Plotting script under util/plot.py. This would help in plotting the embeddings of ID and OOD data.
```
python3 main_with_vis_and_multi_gpu.py --model_name_or_path roberta-large --loss custom --task_name 20ng --plot true
```
If cuda out of memory error occurs, please reduce the batch size using --batch_size argument in the command.

# Authors

- Omkar Chandrakant Prabhune (oprabhune@wisc.edu)
- Sourav Suresh (sourav.suresh@wisc.edu)
