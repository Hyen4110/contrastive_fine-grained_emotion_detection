import os
import math
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, tqdm_notebook

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

def get_labeled_data(data, not_use_emotion=[], save_yn=False, save_path=None):
    ''' input : list(emotion classes) -> 기쁨, 당황, 분노, 불안, 슬픔, 상처
        output : labeled df '''
    le = LabelEncoder()

    if not not_use_emotion:
        data['labels'] = le.fit_transform(data['emotion'])
        file_nm = 'data_class_6(all).csv'
    else:
        for not_emot in  not_use_emotion:
            data = data[data.emotion!=not_emot]
            data.reset_index(drop=True, inplace =True)

        data['labels'] = le.fit_transform(data['emotion'])
        file_nm = 'data_class_'+str(6-len(not_use_emotion))+'.csv'
         
    if save_yn:
        data.to_csv(save_path+'/'+file_nm, index = False)
        print(f"save the dataframe in dir : {save_path+'/'+file_nm}")
    else:
        print(f"not save the dataframe just return")
    return data

# df -> kobert_embeddings -> pickle (: ./text_embeddings)

# 논문의 token max_len =  512/ 지금 데이터의 max_len = 57 
def sentence_to_embedding(model, sentences_list, MAX_LEN=57):
    ''' 문장을 pretrained_KoBERT embeddig으로 변환
        - input : sentences(list)
        - otuput : tensor (batch_size, sequence_length, embedding_dim) '''
    
    inputs = tokenizer(sentences_list, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length = MAX_LEN) 
    # print(inputs.keys())  #['input_ids', 'token_type_ids', 'attention_mask']
    out = model(input_ids = torch.tensor(inputs['input_ids']),
                attention_mask = torch.tensor(inputs['attention_mask']))
    kobert_embeddings = out.last_hidden_state
    # print(kobert_embeddings.shape) # 
    return kobert_embeddings

def kobert_emb_to_pickle_bundle(kobert_embeddings, idx=0, tot=1, save_cnt=None):
    # save data as pickle 
    with open('./data/text_embeddings/kobert_emb_'+str(save_cnt)+'_'+str(idx)+'.pickle','wb') as fw:
            pickle.dump(kobert_embeddings, fw)
    print(f"----[{idx}/{tot}] finished -> \'{'kobert_emb_'+str(save_cnt)+'_'+str(idx)+'.pickle'}\' | {kobert_embeddings.size()}")
    

def text_to_emb_pickle_bundle(model, df, colnm='text', save_cnt = 500):
    # from df get embeddings (save_cnt, MAX_LEN, 768) and save as pickle 
    prev_idx_ = 0
    for i in tqdm(range(1, len(df)//save_cnt+2)):
        idx_ = save_cnt*i if i!=len(df)//save_cnt+1 else None
        temp = list(df[colnm][prev_idx_ : idx_].values)
        # print(temp)
        kobert_emb_to_pickle_bundle(sentence_to_embedding(model, temp), idx=i, tot = len(df)//save_cnt+1, save_cnt = save_cnt)
        prev_idx_ = idx_

def kobert_emb_to_pickle_one(kobert_embeddings, filenm):
    # save data as pickle 
    with open('./data/text_embeddings/kobert_emb_'+filenm+'.pickle','wb') as fw:
            pickle.dump(kobert_embeddings, fw)
    # print(f"\r----finished -> \'{'kobert_emb_'+filenm+'.pickle'}\' | {kobert_embeddings.size()}")
    

def text_to_emb_pickle_one(model, df):
    # from df get embeddings (save_cnt, MAX_LEN, 768) and save as pickle 
    for i in tqdm(range(len(df)), leave=True):
        sentence = [df['text'][i]]
        filenm = df['filenm'][i]
        # print(temp)
        kobert_emb_to_pickle_one(sentence_to_embedding(model, sentence), filenm)

if __name__ == '__main__':
    data = pd.read_csv("./data/texts.csv", encoding='cp949') 
    data = data[['filenm','emotion', 'text']]
    # data = get_labeled_data(data, not_use_emotion=['상처'], save_yn=True, save_path=".")
    data = get_labeled_data(data, not_use_emotion=[], save_yn=True, save_path=".")

    # /text_embeddings/ 폴더 내에 저장! 하나만 선택!

    # 1. bundle로 저장
    # text_to_emb_pickle_bundle(model, df = data[:3]) # test
    # text_to_emb_pickle_bundle(model, df = data) # real


    # 2.하나씩 저장
    text_to_emb_pickle_one(model, df = data[:7], colnm = 'text', save_cnt = 2) # test
    # text_to_emb_pickle_one(model, df = data, colnm = 'text', save_cnt = 500) # real