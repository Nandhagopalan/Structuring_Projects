import utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import string

def loss_fun(o1,o2,t1,t2):
    loss1=nn.BCEWithLogitsLoss()(o1,t1)
    loss2=nn.BCEWithLogitsLoss()(o2,t2)

    return loss1+loss2


def train_fn(data_loader,model,optimizer,device,scheduler):
    model.train()
    losses=utils.AverageMeter()
    tk0=tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["masks"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1,o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(o1,o2,targets_start,targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader,model,device):
    model.eval()
    tk0=tqdm(data_loader, total=len(data_loader))
    
    fin_outputs_start=[]
    fin_outputs_end=[]
    fin_padding_lens=[]
    fin_tweet_tokens=[]
    fin_org_tweet=[]
    fin_org_selected=[]
    fin_org_sentiment=[]

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["masks"]
        tweet_tokens=d['tweet_tokens']
        padding_len=d['padding_len']
        org_sentiment=d['org_sentiment']
        org_selected=d['org_selected_text']
        org_tweet=d['original_tweet']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        

        optimizer.zero_grad()
        o1,o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )


        fin_outputs_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_org_sentiment.extend(org_sentiment)
        fin_org_selected.extend(org_selected)
        fin_org_tweet.extend(org_tweet)

    fin_outputs_start=np.vstack(fin_outputs_start)
    fin_outputs_end=np.vstack(fin_outputs_end)

    threshold=0.2
    jaccards=[]

    for j in range(len(fin_tweet_tokens)):
        target_string=fin_org_selected[j]
        tweet_tokens=fin_tweet_tokens[j]
        padding_len=fin_padding_lens[j]
        original_tweet=fin_org_tweet[j]
        sentiment=fin_org_sentiment[j]

        if padding_len>0:
            mask_start=fin_outputs_start[j,:][:-padding_len]>=threshold
            mask_end=fin_outputs_end[j,:][:-padding_len]>=threshold
        else:
            mask_start=fin_outputs_start[j,:]>=threshold
            mask_end=fin_outputs_end[j,:]>=threshold

        mask=[0]*len(mask_start)

        idx_start=np.nonzero(mask_start)[0]
        idx_end=np.nonzero(mask_end)[0]


        if len(idx_start)>0:
            idx_start=idx_start[0]

            if len(idx_end)>0:
                idx_end=idx_end[0]
            else:
                idx_end=idx_start

        else:
            idx_start=0
            idx_end=0

        for mj in range(idx_start,idx_end+1):
            mask[mj]=1

        output_tokens=[x for p,x in enumerate(tweet_tokens.split()) if mask[p]==1]
        output_tokens=[x for x in output_tokens if x not in ('[CLS]','[SEP]')]

        final_output=''

        for ot in final_output:
            if ot.startswith('##'):
                final_output+=ot[2:]
            elif len(ot)==1 and ot==string.punctuation:
                final_output+=ot
            else:
                final_output= final_output+' '+ot

        final_output=final_output.strip()

        if sentiment=="neutral" or len(original_tweet.split())<=5:
            final_output=original_tweet

        jac=utils.jaccard(final_output.strip(),target_string.strip())
        jaccards.append(jac)

    mean_jac=np.mean(jaccards)
    return mean_jac
    
