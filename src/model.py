import numpy as np 
import torch
from torch import nn
from transformers import BertConfig, BertModel

class KoBERT(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, config, dropout_p=.3):
        super().__init__()
        
        bert_config = BertConfig.from_pretrained('skt/kobert-base-v1', output_hidden_states=True)
        self.kobert = BertModel.from_pretrained('skt/kobert-base-v1', config = bert_config)        
        
        self.embedder = nn.Sequential(nn.LayerNorm(hidden_size),
                                      nn.Linear(hidden_size, embedding_size),
                                      nn.Tanh(),
                                      nn.Dropout(dropout_p)
                                      )
        
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_size),
                                        nn.Linear(embedding_size, output_size),
                                        nn.Tanh(),
                                        nn.Dropout(dropout_p),
                                        nn.Softmax(dim=-1))
        # freeze params
        if config.freeze_yn:
            for param in self.kobert.parameters():
                param.requires_grad = False
           
    def forward(self, token_ids, attention_mask, token_type_ids):    
        outputs = self.kobert(input_ids = token_ids, 
                              token_type_ids = token_type_ids.long(), 
                              attention_mask = attention_mask.float().to(token_ids.device))

        # Mean-Pooling
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, 1)
        embedidngs = self.embedder(pooled_output)
        logits = self.classifier(embedidngs)

        return embedidngs, logits