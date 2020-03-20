import random
import  numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

def get_cat_cont_feats(df,contfeats):
    
    categorical = df.drop(['target'] + contfeats,
                          axis=1).columns

    return categorical,contfeats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministick = True
    torch.backends.cudnn.benchmark = False 


def split_dataset(trainset, valid_size=0.2, batch_size=64):
    num_train = len(trainset)
    
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    split = int(np.floor(valid_size * num_train))
    
    valid_idx, train_idx = indices[:split], indices[split:]
    
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    
    valid_loader = DataLoader(trainset, 
                              batch_size=batch_size, 
                              sampler=valid_sampler)
    train_loader = DataLoader(trainset, 
                              batch_size=batch_size, 
                              sampler=train_sampler)
    
    return train_loader, valid_loader


def cat_dim(all_df,cats):
    
    embedding_cardinality = {n: c.nunique()+1 for n,c in all_df[cats].items()}
    emb_sizes = [(size, max(5, size//2)) for item, size in embedding_cardinality.items()]

    return emb_sizes

    