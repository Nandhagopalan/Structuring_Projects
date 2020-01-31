import torch.nn as nn
import transformers
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AdamW,get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm    #TPU Config
import torch_xla.distributed.xla_multiprocessing as xmp  #multi core
import torch_xla.distributed.parallel_loader as pl       #multi core
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BERTBaseUncased(nn.Module):
    
    def __init__(self,bert_path):

        super(BERTBaseUncased,self).__init__()
        self.bert_path=bert_path
        self.bert=transformers.from_pretrained(self.bert_path)
        self.bert_drop=nn.Dropout(0.3)
        self.out=nn.Linear(768,30)

    def forward(self,ids,masks,token_type_ids):
        _,o2=self.bert(ids,attention_mask=masks,token_type_ids=token_type_ids)
        bo=self.bert_drop(o2)
        return self.out(bo)

    
class BERTDatasetTraining():

    def __init__(self,qtitle,qbody,answer,targets,tokenizer,max_len):
        self.qtitle=qtitle
        self.qbody=qbody
        self.answer=answer
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.targets=targets

    def __len__(self):
        return len(self.answer)

    def __getitem__(self,item):
        question_title=str(self.qtitle[item])
        question_body=str(self.qbody[item])
        answer=str(self.answer[item])

    
        inputs=self.tokenizer.encode_plus(                ## [CLS] + [QT]+[QB] + [SEP] + [A] +[SEP]
            question_title + " " + question_body,
            answer,
            add_special_tokens=True,
            max_length=self.max_len
        )

        ids=inputs['input_ids']
        token_type_ids=inputs['token_type_ids']
        masks=inputs['attention_mask']

        padding_len=self.max_len-len(ids)

        ids= ids+([0]*padding_len)
        token_type_ids = token_type_ids +([0]*padding_len)
        masks = masks + ([0]*padding_len)

        return {
            "ids":torch.tensor(ids,dtype=torch.long),
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),
            "masks": torch.tensor(masks,dtype=torch.long),
            "targets": torch.tensor(self.targets[item,:],dtype=torch.float)
        }


def loss_fn(self,outputs,targets):

    return  nn.BCEWithLogitsLoss()(outputs,targets)

    

def train_loop_fn(self,dataloader,model,optimizer,device,scheduler=None):
    model.train()

    for bid,d in enumerate(dataloader):
        ids=d["ids"]
        token_type_ids=d["token_type_ids"]
        masks=d["masks"]
        targets=d["targets"]

        ids=ids.to(device,dtype=torch.long)
        token_type_ids=token_type_ids.to(device,dtype=torch.long)
        masks=masks.to(device,dtype=torch.long)
        targets=targets.to(device,dtype=torch.float)

        xm.optimizer_step(optimizer,barrier=True) #TPU Config for optim #optimizer.zero_grad() - gpu  ##remove barrier for multi core

        outputs=model(ids=ids,mask=masks,token_type_ids=token_type_ids)
        loss=loss_fn(outputs,targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if bid%10==0:
            print(f"Batch{bid} and loss {loss}") ##xm.master_print() - only one core print

        
def eval_loop_fn(self,dataloader,model,device):
    model.eval()

    fin_outputs=[]
    fin_targets=[]

    for bid,d in enumerate(dataloader):
        ids=d["ids"]
        token_type_ids=d["token_type_ids"]
        masks=d["masks"]
        targets=d["targets"]

        ids=ids.to(device,dtype=torch.long)
        token_type_ids=token_type_ids.to(device,dtype=torch.long)
        masks=masks.to(device,dtype=torch.long)
        targets=targets.to(device,dtype=torch.float)

        outputs=model(ids=ids,mask=masks,token_type_ids=token_type_ids)
        loss=loss_fn(outputs,targets)

        fin_outputs.append(outputs.cpu().detach().numpy())
        fin_targets.append(targets.cpu().detach().numpy())

    return np.vstack(fin_outputs),np.vstack(fin_targets)


def run():  ##index - multicore TPU

    MAX_LEN=512
    EPOCHS=2
    BATCH_SIZE=4

    dfx=pd.read_csv('../inputs/train.csv').fillna("none")
    
    df_train,df_valid=train_test_split(dfx,random_state=42,test_size=0.1)

    df_train=df_train.reset_index(drop=True)
    df_valid=df_valid.reset_index(drop=True)

    sample=pd.read_csv('../inputs/sample_submission.csv')
    target_cols=list(sample["qid"].drop(axis=1).columns)

    train_targets=df_train[target_cols].values
    valid_targets=df_valid[target_cols].values

    tokenizer=transformers.BertTokenizer.from_pretrained('../inputs/bert_base_uncased')

    train_dataset=BERTDatasetTraining(
                    qtitle=df_train.question_title.values,
                    qbody=df_train.question_body.values,
                    answer=df_train.answer.values,
                    targets=train_targets,
                    tokenizer=tokenizer,
                    max_len=MAX_LEN
                    )

    ''' ##dist sampler - multicore TPU
    train_sampler=torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),  #no of cores
        rank=xm.get_ordinal(),
        shuffle=True
    )

    '''

    train_data_loader=torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    #sampler=train_sampler
                    shuffle=True
                    )


    valid_dataset=BERTDatasetTraining(
                    qtitle=df_valid.question_title.values,
                    qbody=df_valid.question_body.values,
                    answer=df_valid.answer.values,
                    targets=valid_targets,
                    tokenizer=tokenizer,
                    max_len=MAX_LEN
                    )

    ''' ##dist sampler - multicore TPU
    valid_sampler=torch.utils.data.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),  #no of cores
        rank=xm.get_ordinal()
    )

    '''
    valid_data_loader=torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=BATCH_SIZE,
                    #sampler=valid_sampler
                    shuffle=True
                    )

    device="cuda"
    lr=3e-5  ## 3e-5*xm.xrt_world_size()
    num_train_steps=int(len(train_dataset)/(BATCH_SIZE*EPOCHS)) ## int(len(train_dataset)/BATCH_SIZE/xm.xrt_world_size()*EPOCHS)

    model=BERTBaseUncased('../inputs/bert_base_uncased').to(device)
    optimizer=AdamW(model.parameters,lr=lr)
    scheduler=get_linear_schedule_with_warmup(
                optimizer,num_warmup_steps=0,num_train_steps=num_train_steps
                )

    for epoch in EPOCHS:
        ## para_loader=pl.ParallelLoader(train_data_loader,[device])
        ## pass this to train loop -> para_loader.per_device_loader(device)

        train_loop_fn(train_data_loader,model,optimizer,device,scheduler)

        ## para_loader=pl.ParallelLoader(valid_data_loader,[device])
        ## pass this to eval loop -> para_loader.per_device_loader(device)

        o,t=eval_loop_fn(valid_data_loader,model,device)

        spear=[]

        for jj in range(t.shape[1]):
            p1=list(t[:,jj])
            p2=list(o[:,jj])

            coef,_=np.nan_num(stats.spearmanr(p1,p2))

            spear.append(coef)

        spear=np.mean(spear)

        print(f"Epoch={epoch}  and spearman is {spear}")  ##xm.master_print() - only one core print

        xm.save(model.state_dict(),"model.bin")  #TPU Config for saving


if __name__=="__main__":
    run()  ## xmp.spawn(run,nprocs=8) - nprocs - no of cores





