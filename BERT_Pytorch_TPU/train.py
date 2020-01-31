import torch.nn as nn
import transformers
import torch

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
            max_len=self.max_len
        )

        ids=inputs['input_ids']
        token_type_ids=inputs['token_type_ids']
        masks=inputs['attention_mask']

        padding_len=self.max_len-len(ids)

        ids= ids+([0]*padding_len)
        token_type_ids = token_type_ids +([0]*padding_len)
        masks = masks + ([0]*padding_len)

        return {
            "ids":torch.tensor(ids,dtype=torch.long)
            "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long)
            "masks": torch.tensor(self.targets[item,:],dtype=torch.float)
        }


def loss_fn(self,outputs,targets):

    return  nn.BCEWithLogitsLoss()(outputs,targets)

    