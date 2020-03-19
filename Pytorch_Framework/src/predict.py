import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def predict(data_loader, model):
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    with torch.no_grad():
        predictions = None
        
        for i, batch in enumerate(tqdm(data_loader)):   
            
            output = model(batch['data'][0].to(device, 
                                               dtype=torch.long), 
                           batch['data'][1].to(device, 
                                               dtype=torch.float)).cpu().numpy()
            
            if i == 0:
                predictions = output
                
            else: 
                
                predictions = np.vstack((predictions, output))
                
    return predictions


def predict_pipe(model,test_dataset):

    #Testing
    model.load_state_dict(torch.load('simple_nn.pt'))

    test_loader = DataLoader(test_dataset, 
                         batch_size=128)

    predictions = predict(test_loader, model) 