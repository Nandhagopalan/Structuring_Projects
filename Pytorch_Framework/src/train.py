import pandas as pd
import warnings
from featuregenerator import Featureengineering
from utils import get_cat_cont_idx,set_seed,split_dataset,cat_dim
from dataset import ClassificationDataset
from model import ClassificationEmbdNN
from engine import train_network
import torch
from predict import predict_pipe


warnings.filterwarnings('ignore')


def run():
    train=pd.read_csv('../input/train.csv')
    test=pd.read_csv('../input/test.csv')

    traintest=pd.concat([train,test],ignore_index=True)
    traintest=traintest.drop('id',axis=1)

    feateng=Featureengineering(traintest,cyc_feats=['month','day'],bin_feats=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'],
                                ord_feats=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'],nom_feats=['nom_0', 'nom_1',
                                'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])

    continuous=['month_sin', 'month_cos','day_sin', 'day_cos']

    dataset=feateng.engineer()

    train=dataset[dataset['target']!=-1]
    test=dataset[dataset['target']==-1]

    
    traindata=train.drop(['target'],axis=1).to_numpy()
    target=train['target'].to_numpy()

    testdata=test.drop(['target'],axis=1).to_numpy()


    print(traindata.shape,testdata.shape)

    cat_idx,cont_idx=get_cat_cont_idx(train,continuous)

    set_seed(2020)

    train_dataset = ClassificationDataset(traindata, 
                                      targets=target,
                                      cat_cols_idx=cat_idx,
                                      cont_cols_idx=cont_idx)
    test_dataset = ClassificationDataset(testdata,
                                     cat_cols_idx=cat_idx,
                                     cont_cols_idx=cont_idx,
                                     is_train=False)

    train_loader, valid_loader = split_dataset(train_dataset, 
                                           valid_size=0.2,batch_size=2000) 

    cat_dims=cat_dim(train,continuous)   

    model = ClassificationEmbdNN(emb_dims=cat_dims, 
                             no_of_cont=len(continuous))

    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    train_losses, valid_losses = train_network(model=model, 
                                           train_loader=train_loader, 
                                           valid_loader=valid_loader, 
                                           loss_func=loss_func, 
                                           optimizer=optimizer,
                                           n_epochs=3, 
                                           saved_model='simple_nn.pt')     

    return model,test_dataset
    
if __name__=='__main__':
    model,test=run()
    inference=True
    if inference:
        predictions=predict_pipe(model,test)
        sumbission=pd.read_csv('../input/sample_submission.csv')
        nn_predictions_df = pd.DataFrame({'id': sumbission['id'], 'target': predictions.squeeze()})
        nn_predictions_df.to_csv('submission.csv',index=None)
