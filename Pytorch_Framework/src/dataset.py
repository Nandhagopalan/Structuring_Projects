import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, data, targets=None,
                 is_train=True, cat_cols_idx=None,
                 cont_cols_idx=None):
        self.data = data
        self.targets = targets
        self.is_train = is_train
        self.cat_cols_idx = cat_cols_idx
        self.cont_cols_idx = cont_cols_idx
    
    def __getitem__(self, idx):
        row = self.data[idx].astype('float32')
        
        data_cat = []
        data_cont = []
        
        result = None
        
        if self.cat_cols_idx:
            data_cat = torch.tensor(row[self.cat_cols_idx])
            
        if self.cont_cols_idx:
            data_cont = torch.tensor(row[self.cont_cols_idx])
                
        data = [data_cat, data_cont]
                
        if self.is_train:
            result = {'data': data,
                      'target': torch.tensor(self.targets[idx])}
        else:
            result = {'data': data}
            
        return result
        
  
    
    def __len__(self):
        return(len(self.data))

