import random
import  numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

def get_cat_cont_idx(df,contfeats):
    
    categorical = df.drop(['target'] + contfeats,
                          axis=1).columns
    
    cat_cols_idx, cont_cols_idx = list(), list()
    df=df.drop(['target'],axis=1)

    for idx, column in enumerate(df.columns):

        if column in categorical:
            cat_cols_idx.append(idx)
        elif column in contfeats:
            cont_cols_idx.append(idx)

    return cat_cols_idx,cont_cols_idx


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


def cat_dim(all_df,cont_feats):
    
    categorical = all_df.drop(['target'] + cont_feats,
                          axis=1).columns

    cat_dim = [int(all_df[col].nunique()) for col in categorical]
    cat_dim = [[x, min(200, (x + 1) // 2)] for x in cat_dim]

    for el in cat_dim:
        if el[0] < 10:
            el[1] = el[0]

    return cat_dim