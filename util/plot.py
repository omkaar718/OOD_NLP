from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
from util.analysis import get_embeddings
import numpy as np
from statistics import mean, median

def plot(args, model, dataset, ood_dataset, task_name, num_categories = 6):

    in_embeddings, test_targets = get_embeddings(args, model, dataset)
    ood_embeddings, _ = get_embeddings(args, model, ood_dataset, is_id=False)
    tsne = TSNE(2, verbose=1)

    embeddings = np.array(in_embeddings.cpu())
    test_targets = np.array(test_targets.cpu())
    tsne_proj = tsne.fit_transform(embeddings)
    
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    
    for lab in range(num_categories):
        indices = test_targets==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.scatter(np.array(mean(tsne_proj[indices,0])),np.array(mean(tsne_proj[indices,1])), color="black", marker = 'x' ,alpha=0.5)

    ood_embeddings = np.array(ood_embeddings.cpu())
    tsne_proj = tsne.fit_transform(ood_embeddings)
    ax.scatter(tsne_proj[:,0],tsne_proj[:,1], color="black", marker = ".", label = "ood" ,alpha=0.5)


    ax.legend(fontsize='large', markerscale=2)
    plt.savefig(f"{task_name}_tnse.png")

