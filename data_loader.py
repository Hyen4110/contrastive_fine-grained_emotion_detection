import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import Wav2Vec2FeatureExtractor
import argparse
import random
# import librosa 
import torchaudio
from KoBERT.kobert_hf.kobert_tokenizer import KoBERTTokenizer
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class FreezedDataset(Dataset):
    '''
    data(dataframe)
    | --- |  filenm   | label |
    |  1  |  M_00100  |   0   |
    '''
    def __init__(self, config, data):
        self.config = config
        self.data = data
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filenm = self.data['filenm'].loc[idx]
        text = self.load_text_emb(filenm) # (bs, 57, 768)
        audio = self.load_audio_emb(filenm) # (bs, 128, 1100)
        label = self.data['labels'].loc[idx]
        return {'text': text, 'audio' :audio, 'label': label}

    def load_text_emb(self, filenm):
        full_path = self.config.path_text + '/' + self.config.pre_text + filenm + '.pickle' # kobert_emb_F_000085.pickle
        text_embedding = pickle.load(open(full_path, 'rb')) 
        return text_embedding

    def load_audio_emb(self, filenm):
        full_path = self.config.path_audio + '/' + self.config.pre_audio + filenm + '.pickle' # mfcc_F_000085.pickle
        audio_embedding = pickle.load(open(full_path, 'rb'))
        return audio_embedding
    
class FineTuneDataset(Dataset):

    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', 
                                                        sp_model_kwargs={'nbest_size': -1, 
                                                        'alpha': 0.6, 
                                                        'enable_sampling': True})
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filenm = self.data['filenm'].loc[idx]        
        text = self.text_to_inputids(self.data['text'].loc[idx])
        
        label = self.data['labels'].loc[idx]
        label = torch.tensor(label, dtype=torch.long)
        return {'text': text, 'label': label}
        
    def load_text_emb(self, filenm):
        full_path = self.config.path_text + '/' + self.config.pre_text + filenm + '.pickle' # kobert_emb_F_000085.pickle
        text_embedding = pickle.load(open(full_path, 'rb'))
        return text_embedding
    
    def text_to_inputids(self, text):
        MAX_LEN = 57
        inputs = self.tokenizer(text, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length = MAX_LEN)
        
        token_ids = inputs['input_ids'] # tensor type
        attention_mask = inputs['attention_mask'] # tensor type
        token_type_ids = inputs['token_type_ids'] # tensor type
        # print(f"inputs['input_ids'] type : {type(inputs['input_ids'])}")
        
        return [token_ids, attention_mask, token_type_ids]
    
def load_data(config):
    data = pd.read_csv(config.data_path)
    print(data.info())
    # data = data[['filenm','labels']]
        
    return data

# split into train/valid/test
def get_data_split(config, test = True):
    # step 1 -> train | valid + test
    all_data = load_data(config)
    X_, y_ = all_data.filenm.values, all_data.labels.values
    
    if test == True:    
        train_split = StratifiedShuffleSplit(n_splits=1, test_size= 0.3, random_state=0)
        for train_idx_, valtest_idx_ in train_split.split(X_, y_):
            train_idx,  valtest_idx = train_idx_, valtest_idx_
        
        # stepp 2 -> valid | test
        val_test_data = all_data.loc[valtest_idx].reset_index(drop=True)
        X_, y_ = val_test_data.filenm.values, val_test_data.labels.values
        val_test_split = StratifiedShuffleSplit(n_splits=1, test_size= 0.3, random_state=0)
        for valid_idx_, test_idx_ in val_test_split.split(X_, y_):
            valid_idx, test_idx = valid_idx_, test_idx_
        
        train_df = all_data.loc[train_idx].reset_index(drop=True)
        valid_df = val_test_data.loc[valid_idx].reset_index(drop=True)
        test_df =  val_test_data.loc[test_idx].reset_index(drop=True)
        print(f" train: valid: test = {len(train_df)/len(all_data):.1f} : {len(valid_df)/len(all_data):.1f} : {len(test_df)/len(all_data):.1f}")
        return train_df, valid_df, test_df
    
    else :
        X_, y_ = all_data.filenm.values, all_data.labels.values
        train_test_split = StratifiedShuffleSplit(n_splits=1, test_size= 0.1, random_state=0)
        for train_idx_, valid_idx_ in train_test_split.split(X_, y_):
            train_idx,  valid_idx = train_idx_, valid_idx_
        
        train_df = all_data.loc[train_idx].reset_index(drop=True)
        valid_df = all_data.loc[valid_idx].reset_index(drop=True)
        return train_df, valid_df
        
def get_loaders(config):
    train_df, valid_df, test_df = get_data_split(config)
    train_loader = DataLoader( dataset = FineTuneDataset( config, train_df),
                                batch_size = config.batch_size,
                                shuffle=True)
    valid_loader = DataLoader( dataset = FineTuneDataset( config, valid_df),
                                batch_size=config.batch_size,
                                shuffle=True)
    test_loader = DataLoader( dataset = FineTuneDataset( config, test_df),
                                batch_size=config.batch_size,
                                shuffle=False)
                                
    config.n_train, config.n_valid, config.n_test =len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)
    print(f" [data counts] train_loader: {len(train_loader.dataset)}, valid_loader : {len(valid_loader.dataset)}, test_loader : {len(test_loader.dataset)}")
    
    return train_loader, valid_loader, test_loader