import torch


class ClassificationEmbdNN(torch.nn.Module):
    
    def __init__(self, emb_dims, no_of_cont=None):
        super(ClassificationEmbdNN, self).__init__()
        
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(x, y)
                                               for x, y in emb_dims])
        
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.emb_dropout = torch.nn.Dropout(0.2)
        
        self.no_of_cont = 0
        if no_of_cont:
            self.no_of_cont = no_of_cont
            self.bn_cont = torch.nn.BatchNorm1d(no_of_cont)
        
        self.fc1 = torch.nn.Linear(in_features=self.no_of_embs + self.no_of_cont, 
                                   out_features=256)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.act1 = torch.nn.ReLU()
        
        self.fc2 = torch.nn.Linear(in_features=256, 
                                   out_features=256)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.act2 = torch.nn.ReLU()
        
        self.fc3 = torch.nn.Linear(in_features=256, 
                                   out_features=64)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.act3 = torch.nn.ReLU()
        
        self.fc4 = torch.nn.Linear(in_features=64, 
                                   out_features=1)
        self.act4 = torch.nn.Sigmoid()
        
    def forward(self, x_cat, x_cont=None):
        if self.no_of_embs != 0:
            x = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        
            x = torch.cat(x, 1)
            x = self.emb_dropout(x)
            
        if self.no_of_cont != 0:
            x_cont = self.bn_cont(x_cont)
            
            if self.no_of_embs != 0:
                x = torch.cat([x, x_cont], 1)
            else:
                x = x_cont
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        x = self.fc4(x)
        x = self.act4(x)
        
        return x