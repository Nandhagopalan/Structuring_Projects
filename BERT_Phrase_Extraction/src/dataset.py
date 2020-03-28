import config
import torch
import numpy as np

class TweetDataset:

    def __init__(self,tweet,selected_text,sentiment):
        self.tweet=tweet
        self.selected_text=selected_text
        self.sentiment=sentiment
        self.max_length=config.MAX_LEN
        self.tokenizer=config.TOKENIZER


    def __len__(self):
        return len(self.tweet)

    def __getitem__(self,item):
        tweet=" ".join(str(self.tweet[item]).split())
        selected_text=" ".join(str(self.selected_text[item]).split())

        sel_len=len(selected_text)

        idx0=-1
        idx1=-1

        for ind in (i for i,e in enumerate(tweet) if e==selected_text[0]):
            if tweet[ind:ind+sel_len]==selected_text:
                idx0=ind
                idx1=ind+sel_len-1
                break

        char_targets=[0]*len(tweet)

        if idx0!=-1 and idx1!=-1:
            for j in range(idx0,idx1+1):
                if tweet[j]!=' ':
                    char_targets[j]=1
        
        tok_tweet=self.tokenizer.encode(tweet)

        tok_tweet_tokens=tok_tweet.tokens
        tok_tweet_ids=tok_tweet.ids
        tok_tweet_offsets=tok_tweet.offsets[1:-1]

        targets=[0]*(len(tok_tweet_tokens)-2)

        for j,(offset1,offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2])>0:
                targets[j]=1

        targets=[0]+targets+[0]

        targets_start=[0]*len(targets)
        targets_end=[0]*len(targets)

        non_zero=np.nonzero(targets)[0]

        if len(non_zero)>0:
            targets_start[non_zero[0]]=1
            targets_end[non_zero[-1]]=1

        
        mask=[1]*len(tok_tweet_ids)
        token_type_ids=[0]*len(tok_tweet_ids)

        padding_len=self.max_length-len(tok_tweet_tokens)

        ids=tok_tweet_ids+[0]*padding_len
        masks=mask+[0]*padding_len
        token_type_ids=token_type_ids+[0]*padding_len
        targets=targets+[0]*padding_len
        targets_start=targets_start+[0]*padding_len
        targets_end=targets_end+[0]*padding_len

        sentiment=[1,0,0]

        if self.sentiment[item]=='positive':
            sentiment=[0,0,1]
        elif self.sentiment[item]=='negative':
            sentiment=[0,1,0]


        return {
            "ids":torch.tensor(ids,dtype=torch.long),
            "masks":torch.tensor(masks,dtype=torch.long),
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),
            "targets":torch.tensor(targets,dtype=torch.long),
            "targets_start":torch.tensor(targets_start,dtype=torch.long),
            "targets_end":torch.tensor(targets_end,dtype=torch.long),
            "padding_len":torch.tensor(padding_len,dtype=torch.long),
            "tweet_tokens":" ".join(tok_tweet_tokens),
            "original_tweet":self.tweet[item],
            "sentiment":torch.tensor(sentiment,dtype=torch.long),
            "org_sentiment":self.sentiment[item],    
            "org_selected_text":self.selected_text[item],
            
        }


# a=TweetDataset(['i love my youtube channel!'],['my youtube'],['positive'])

# print("fff",a[0])