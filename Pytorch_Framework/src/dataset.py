import torch
from torch.utils.data import Dataset
import numpy as np


class ClassificationDataset(Dataset):
    def __init__(self, data, targets=None,
                 is_train=True, cat_cols=None,
                 cont_cols=None):
        self.data = data
        self.targets = targets
        self.is_train = is_train

        cat_values = [c.values for n,c in self.data[cat_cols].items()]
        cont_values = [c.values for n,c in self.data[cont_cols].items()]

        self.cat_features = np.stack(cat_values, 1).astype(np.int64)
        self.cont_features = np.stack(cont_values, 1).astype(np.float32)
    
    def __getitem__(self, idx):

        cat_val = self.cat_features[idx]
        cont_val = self.cont_features[idx]

        result = None
                     
        data = [cat_val, cont_val]
                
        if self.is_train:
            result = {'data': data,
                      'target': torch.tensor(self.targets[idx],dtype=float)}
        else:
            result = {'data': data}
            
        return result  
  
    
    def __len__(self):
        return(len(self.data))
