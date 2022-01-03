import random
import argparse
import pprint
import torch

import train
from data_loader import get_loaders

ver_description = {2: "KoBERT cross-entropy",
                   3: "KoBERT contrastive",
                   4: "KoBERT(freeze) contrastvie"
                   }

def define_argparser():
    parser = argparse.ArgumentParser(description='Korean text fine-grained emotion deteciton')

    ## version (model)
    parser.add_argument('--version', type=int, default = 2)
    parser.add_argument('--n_class', type=int, default = 6) 
    parser.add_argument('--freeze_yn', type=bool, default = False) # not hyper-parameter

    parser.add_argument('--embedding_size', type=int, default = 128)  
    parser.add_argument('--hidden_size', type=int, default = 128)

        
    ## training details
    parser.add_argument('--num_epochs', type=int, default = 20)
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--clip', type=float, default = 8.0)
    parser.add_argument('--learning_rate', type=float, default= 1e-4)
    parser.add_argument('--dropout_p', type=float, default= 0.3) 

    ## hyper-parameter of contrastive  learning
    parser.add_argument('--dist_metric', type=str, default= 'euc',
                    help='euclidean-distance / cosine-similarity ')
    # loss ratio
    parser.add_argument('--r_ce', type=float, default= 0.5,
                        help='ratio of cross-entropy')
    parser.add_argument('--r_trplt', type=float, default= 0.3,
                        help='ratio of triplet loss')
    parser.add_argument('--r_arcf', type=float, default= 17.1,
                        help='ratio of arcface loss')
    # margin
    parser.add_argument('--m_trplt', type=float, default= 0.3,
                        help='margin for triplet loss')
    parser.add_argument('--m_arcf', type=float, default= 0.3,
                        help='margin for arcface loss')
    
    # data path
    parser.add_argument('--data_path', type=str, default='./data/data_class_6(all).csv',
                        help='path for raw text data')  # './data/data_class_6(all).csv   or data_class_5.csv'
    parser.add_argument('--path_text', type=str, default='./data/text_embeddings_kobert',
                        help='path for freezed text embeddings pickles')
    parser.add_argument('--pre_text', type=str, default='kobert_emb_',
                        help='added words before the text file name(ids)')
    
    config = parser.parse_args()
    return config

def print_config(config):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(config))


# set random seed
torch.manual_seed(config.seed) # for CPU
torch.cuda.manual_seed(config.seed) # for CUDA
random.seed(config.seed) 
torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == '__main__':
    config = define_argparser() 
    print_config(config)

    train_loader, valid_loader, test_loader = get_loaders(config)
    train.initiate(config, train_loader, valid_loader, test_loader)