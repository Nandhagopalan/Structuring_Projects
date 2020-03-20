import torch
from tqdm import tqdm
from sklearn import metrics
import  numpy as np

def train_network(model, train_loader, valid_loader,
                  loss_func, optimizer, n_epochs=20,
                  saved_model='model.pt'):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses = list()
    valid_losses = list()
    
    valid_loss_min = np.Inf
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        train_auc = 0.0
        valid_auc = 0.0
        
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            output = model(batch['data'][0].to(device, 
                                               dtype=torch.long),
                           batch['data'][1].to(device, 
                                               dtype=torch.float))
            

            loss = loss_func(output, batch['target'].to(device,dtype=torch.float))

            loss.backward()
            optimizer.step()
            
            train_auc += metrics.roc_auc_score(batch['target'].cpu().numpy(),
                                               output.detach().cpu().numpy())

            train_loss += loss.item() * batch['data'][0].size(0) 
    

        model.eval()

        for batch in tqdm(valid_loader):
            output = model(batch['data'][0].to(device, 
                                               dtype=torch.long),
                           batch['data'][1].to(device, 
                                               dtype=torch.float))
            
           
            loss = loss_func(output, batch['target'].to(device,dtype=torch.float))
            
            valid_auc += metrics.roc_auc_score(batch['target'].cpu().numpy(),
                                               output.detach().cpu().numpy())
            valid_loss += loss.item() * batch['data'][0].size(0)  #!!!
           
        
        train_loss = np.sqrt(train_loss / len(train_loader.sampler.indices))
        valid_loss = np.sqrt(valid_loss / len(valid_loader.sampler.indices))

        train_auc = train_auc / len(train_loader)
        valid_auc = valid_auc / len(valid_loader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {}. Training loss: {:.6f}. Validation loss: {:.6f}'
              .format(epoch, train_loss, valid_loss))
        print('Training AUC: {:.6f}. Validation AUC: {:.6f}'
              .format(train_auc, valid_auc))
        
        if valid_loss < valid_loss_min:  # let's save the best weights to use them in prediction
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model...'
                  .format(valid_loss_min, valid_loss))
            
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss
            
    
    return train_losses, valid_losses
        