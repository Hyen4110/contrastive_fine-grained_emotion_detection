import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_centor_distance(config, embeddings, labels, state = 'train', dist = 'euc'):
    num_class = config.n_class
    
    df = pd.DataFrame(embeddings)    
    df['label'] = labels
    groupby_label = df.groupby(['label']).mean()
    label_center = { i: torch.tensor(groupby_label.loc[i].values) for i in range(num_class)} # center of label_center (dim : 128)
    
    if dist == 'cos':
        compute_dist= torch.nn.CosineSimilarity(dim=0)
    else:
        compute_dist = torch.nn.PairwiseDistance(p=2)
    
    dist_df = pd.DataFrame(index = list(range(num_class)), columns = list(range(num_class)))
    for i in range(num_class):
        for j in range(i,num_class):
            dist_df[i][j], dist_df[j][i] = [round(compute_dist(label_center[i], label_center[j]).item(),2)]*2
    
    if state !='train':
        print(f"   >> tsne started!! distance type : {dist} ===========================")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(dist_df)

    return label_center
        
        
def plot_t_SNE(config, epoch, embeddings, labels, state = 'train'):
    title_ = 'ver'+str(config.version)+"_"+ state +'_epoch'+str(epoch)+ '_tsne'

    # ############## get accuracy from similarity
    label_center = get_centor_distance(config, embeddings, labels, state, dist = config.dist_metric)
    
    ############# t-SNE
    plt.figure(epoch) # not stacked 
    tsne = TSNE(n_components = 2,
                perplexity = 25,
                random_state = 10)
    tsne_ref = tsne.fit_transform(embeddings)
    
    df = pd.DataFrame()
    df["comp-1"] = tsne_ref[:,0]
    df["comp-2"] = tsne_ref[:,1]
    df['y'] = labels
    
    flatui = ["#071F41", "#004B5A", "#F6C76F", "#ED4534", "#354A21","#7A1712"][:config.n_class]
    ax = sns.scatterplot(x="comp-1", 
                        y="comp-2",
                        hue="y",
                        s = 10,
                        alpha=.7,
                        palette = flatui, 
                        data = df)

    handles, labels = ax.get_legend_handles_labels()
    # customed_label = ['Happy', 'Embarrassed', 'Anger', 'Anxiety', 'Sadness']
    customed_label = ["Happiness", "Embarrassment", "Anger", "Anxiety", "Heartbroken", "Sadness"]
    
    # box 줄이기
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # plt.rcParams["figure.figsize"] = (10,8) # figure 크기 고정
    l = plt.legend(handles[0:config.n_class], customed_label,
                   title = 'Emotions',
                   loc='center left', 
                   bbox_to_anchor=(1, 0.5),
                    ncol = 1 # class 6 # horizental legend
                #    ncol = config.n_class # vertical legend
                #    borderaxespad=0.,
                   )

    plt.savefig(title_+ '.png')
    # print(f"******** finish save tsne plot : {title_}")
    
    return label_center
