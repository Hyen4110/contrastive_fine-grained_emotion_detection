import torch
import numpy as np
from itertools import product

def get_indices_tuple(config, labels):
    device = labels.device
    labels = labels.detach().cpu().numpy()
    num_class = config.output_dim
    
    idx_dict = {key : [] for key in range(num_class)}
    for i in range(num_class):
        idx_dict[i] = list(np.where(labels ==i)[0])

    indices_list = []
    for anc_i in range(num_class):
        ancor = idx_dict[anc_i]
        neg = []
        for neg_i in [i for i in range(5) if i!=anc_i]:
            neg += idx_dict[neg_i]
        indices_list.extend(list(product(*[ancor, ancor, neg])))

    indices_list = [i for i in indices_list if i[0] != i[1]]
    indices_list = [(i[1], i[0], i[2]) for i in indices_list if i[0] > i[1]]

    df = pd.DataFrame(indices_list, columns = ['anc', 'pos', 'neg'])
    df.drop_duplicates(inplace = True)

    ancors = torch.tensor(df.anc.values).long().to(device)
    positives = torch.tensor(df.pos.values).long().to(device)
    negatives = torch.tensor(df.neg.values).long().to(device)
    
    return (ancors, positives, negatives)


def get_accacy_similarity(config, epoch, embeddings, labels, label_center):
        preds = []
        compute_dist = torch.nn.PairwiseDistance(p=2)
        
        for i in tqdm(range(len(labels))):
            row_i = torch.tensor(embeddings[i]) # (1,128)
            distance_from_center = []
            
            for j in range(config.output_dim):
                distance_from_center.append(compute_dist(row_i, label_center[j]).item())
            
            preds.append(np.argmin(distance_from_center))
            
        f1_score_ = f1_score(np.array(labels), np.array(preds), average = 'macro')                   
        accuracy = (np.array(labels) == np.array(preds)).sum().item() / len(labels)
        
        if config.version == 3:
            print(f" ---- Epoch {epoch} Confusion Matrix ----")
            print(confusion_matrix(np.array(labels), np.array(preds)))
            print("\n")
            print(f"[ Epoch {epoch} ]similarity from center accuracy = {accuracy:.4f}, f1_score = {f1_score_:.4f} \n\n")
        print("-"*90)
    

def save_model(config, epoch, model):        
    PATH = './KoBERT_'+str(config.version)+'emb'+str(config.embedding_dim)+'_state'+ str(epoch)+ '.pt'
    torch.save(model.state_dict(), PATH)
