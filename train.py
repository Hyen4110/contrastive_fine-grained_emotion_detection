import torch
from torch import nn

from model import *
from tsne import *
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch_optimizer as custom_optim

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from pytorch_metric_learning import losses, miners, reducers, distances
from utils import * 

####################################################################
#
# set model /criterion/ optimizer
#
####################################################################

def get_model(config):
    config.freeze_yn = True if config.version == 4 else False
    model = KoBERT(embedding_size = config.embedding_size, # sentence embedding dimensionality
                    hidden_size = config.hidden_size, # 768(fixed for kobert)
                    output_size = cofnig.n_class, 
                    config = config,
                    dropout_p = config.dropout_p
                    )
    return model
    
def get_crit(config):
    if config.version == 2: 
        crit_ce = nn.CrossEntropyLoss()
        crits = {'crit_ce': crit_ce}
    else: 
        # set distance 
        if config.dist_metric == 'cos':
            dist_ = distances.CosineSimilarity()
        else:
            dist_ = distances.LpDistance(power = 2)

        crit_ce = nn.CrossEntropyLoss()
        crit_trplt = losses.TripletMarginLoss(margin = config.m_trplt, distance = dist_)  
        crit_arcf = losses.ArcFaceLoss(num_classes = 6, 
                                        embedding_size = config.embedding_size, 
                                        margin = config.m_arcf, 
                                        scale = 8)

        crits = {'crit_ce': crit_ce,
                'crit_trplt': crit_trplt,
                'crit_arcf':crit_arcf}

    return crits

def get_optimizer(config, model):
    scaler = torch.cuda.amp.GradScaler()
    optimizer = custom_optim.RAdam(model.parameters(), lr=config.learning_rate)
    
    return optimizer, scaler

def initiate(config, train_loader, valid_loader, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(config).to(device)
    optimizer, scaler = get_optimizer(config, model).to(device)
    criterion = get_crit(config).to(device)

    settings = {'model': model,
                'scaler' : scaler,
                'optimizer': optimizer,
                'criterion': criterion}
    
    return train_model(settings, config, train_loader, valid_loader, test_loader, scaler)


####################################################################
#
# training and evaluation scripts
#
####################################################################

def train_model(settings, config, train_loader, valid_loader, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = settings['model']
    scaler = settings['scaler']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    def train(epoch, model, optimizer, criterion, scaler):
        train_loss = 0.0
        total_pred, total_label = [], []
        total_embeddings = np.empty((0,config.embedding_size), int)
        num_batches = config.n_train // config.batch_size 
        
        model.train()
        if config.version == 2 :
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                
                    text = batch['text']
                    token_ids = text[0].squeeze(1).long().to(device) # [batch_size, length]
                    attention_mask = text[1].squeeze(1).long().to(device) # [batch_size, length]
                    token_type_ids = text[2].squeeze(1).long().to(device) # [batch_size, length]
                    
                    label = batch['label'].long().to(device).view(-1) # [batch_size ]

                    optimizer.zero_grad()            
                    with torch.cuda.amp.autocast():
                        _, preds = model(token_ids, attention_mask, token_type_ids)
                        loss = criterion['crit_ce'](preds, label)
                    
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss

                    # append- epoch-wise
                    total_pred += torch.argmax(preds, dim=-1).detach().cpu().numpy().tolist()
                    total_label +=  label.detach().cpu().numpy().tolist()

            accuracy = accuracy_score(total_label, total_pred)
            f1_score = f1_score(total_label, total_pred, average = 'macro')
            print(f'[Epoch {epoch} training ] : loss = {float(train_loss/num_batches) :.4f}, accuracy = {accuracy:.4f}, f1_score = {f1_score:.4f}')
                                
            return accuracy, train_loss/num_batches
            
        elif config.version == 3 :
            loss_ce, loss_trplt, loss_arcf = 0.0, 0.0, 0.0
            num_indices = 0
            
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    text = batch['text']
                    token_ids = text[0].squeeze(1).long().to(device) # [128, 57]
                    attention_mask = text[1].squeeze(1).long().to(device) # [128, 57]
                    token_type_ids = text[2].squeeze(1).long().to(device) # [128, 57]

                    label = batch['label'].long().to(device).view(-1) # [128]

                    optimizer.zero_grad()            
                    with torch.cuda.amp.autocast():
                        embeddings, preds  = model(token_ids, attention_mask, token_type_ids)                        
                        indices_tuple = get_indices_tuple(config, label)
                    
                        loss_ce_ = criterion['crit_ce'](preds, label) if config.r_ce != 0 else 0
                        loss_trplt_ = criterion['crit_trplt'](embeddings, label, indices_tuple) if config.r_trplt != 0 else 0
                        loss_arcf_ = criterion['crit_arcf'](embeddings, label, indices_tuple) if config.r_arcf != 0 else 0
                                
                        loss =  loss_ce_*config.r_ce + \
                                loss_trplt_*config.r_trplt + \
                                loss_arcf_*config.r_arcf

                    scaler.scale(loss).backward()                 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # append-epoch-wise
                    ## 1) loss
                    train_loss += loss 
                    loss_ce += loss_ce_ 
                    loss_trplt += loss_trplt_
                    loss_arcf += loss_arcf_
                    num_indices += len(indices_tuple[0])
                    ## 2) output
                    total_embeddings = np.vstack([total_embeddings, embeddings.float().detach().cpu().numpy()])
                    total_pred += torch.argmax(preds, dim=-1).detach().cpu().numpy().tolist()
                    total_label +=  label.detach().cpu().numpy().tolist()

            accuracy = accuracy_score(total_label, total_pred)
            f1_score = f1_score(total_label, total_pred, average = 'macro')

            print(f'[Epoch {epoch} training ] : loss = {float(train_loss/num_batches):.4f}, \
                    accuracy = {accuracy:.4f}, f1_score = {f1_score:.4f},\
                    loss_ce : {loss_ce / num_batches:.4f}, \
                    loss_trplt : {loss_trplt/num_batches:.4f}, \
                    loss_arcface : {loss_arcf /num_batches:.4f}, \
                    num_indices_tuple : {num_indices // num_batches}')
            
            return total_embeddings, total_label
 
    def valid(epoch, model, criterion, test=False):
        loader = test_loader if test else valid_loader

        valid_loss = 0.0
        num_batches = config.n_test // config.batch_size if test else config.n_valid // config.batch_size
        total_embeddings = np.empty((0,config.embedding_size), int)
        total_pred, total_label = [], []

        model.eval()           

        if config.version == 2:
            with torch.no_grad():
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        text = batch['text']
                        token_ids = text[0].squeeze(1).long().to(device) # [128, 57]
                        attention_mask = text[1].squeeze(1).long().to(device) # [128, 57]
                        token_type_ids = text[2].squeeze(1).long().to(device) # [128, 57]

                        label = batch['label'].long().to(device).view(-1) # [128]
                        
                        with torch.cuda.amp.autocast():
                            embeddings, preds = model(token_ids, attention_mask, token_type_ids)
                            loss = criterion['crit_ce'](preds, label)

                        valid_loss += loss
                        
                        # append              
                        total_pred += torch.argmax(preds, dim=-1).detach().cpu().numpy().tolist()
                        total_label += label.detach().cpu().numpy().tolist()
                        
                if test and (epoch >3):
                    print(f" testing Confusion Matrix")
                    print(confusion_matrix(total_label, total_pred))
                f1_score_ = f1_score(total_label, total_pred, average = 'macro')  
                accuracy = accuracy_score(total_label, total_pred)
                print(f"[Epoch {epoch} {'testing' if test else 'validating'}] : loss = {float(valid_loss)/num_batches:.4f}, accuracy = {accuracy:.4f}, f1_score = {f1_score_:.4f}")

            return total_embeddings, total_label
            
        elif config.version in [3,4]:
            
            with torch.no_grad():
                with tqdm(loader, unit="batch") as tepoch:
                    for batch in tepoch:
                        
                        text = batch['text']
                        token_ids = text[0].squeeze(1).long().to(device) # [128, 57]
                        attention_mask = text[1].squeeze(1).long().to(device) # [128, 57]
                        token_type_ids = text[2].squeeze(1).long().to(device) # [128, 57]

                        label = batch['label'].long().to(device).view(-1) # [128]
                        
                        with torch.cuda.amp.autocast():
                            embeddings, preds = model(token_ids, attention_mask, token_type_ids)
                            loss = criterion['crit_ce'](preds, label)

                        valid_loss += loss
                        
                        # append
                        total_embeddings = np.vstack([total_embeddings, embeddings.detach().cpu().numpy()])                        
                        total_pred += torch.argmax(preds, dim=-1).detach().cpu().numpy().tolist()
                        total_label += label.detach().cpu().numpy().tolist()
                                                    
                print(f"[Epoch {epoch} {'testing' if test else 'validating'}] : loss = {float(valid_loss)/num_batches:.4f}")

            return total_embeddings, total_label
                

#################################################################################
#                Let's Start training / validating / testing
#################################################################################

    if config.version == 2 :
        for epoch in range(1, config.num_epochs+1):
            train(epoch, model, optimizer, criterion, scaler) # trian
            valid(epoch, model, criterion, test=False) # valid
            valid(epoch, model, criterion, test=True) # test

            save_model(config, epoch, model)
                
    else:
        for epoch in range(1, config.num_epochs+1):
            # train with tsne
            embeddings, label = train(epoch, model, optimizer, criterion, scaler) # trian
            label_center = plot_t_SNE(config, epoch, embeddings, label, state='train')  # dictionary {label: center(128,)}           
            
            # valid 
            valid(epoch, model, criterion,test=False)
            
            # test with tsne
            embeddings, label = valid(epoch, model, criterion, test=True)
            get_accacy_similarity(config, epoch, embeddings, label, label_center)
            plot_t_SNE(config, epoch, embeddings, label, state='test')
            
            #save model
            save_model(config, epoch, model) # Model Save