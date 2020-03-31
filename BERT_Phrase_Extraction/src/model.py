import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.lo = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        seq_out, pooled_out = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        #seq_out -> batch,tokens,hiddensize
        #pooled -> batch,hiddensize
        
        logits = self.lo(seq_out)
        #(batch,tokens,2)

        start_logits,end_logits=logits.split(1,dim=-1)
        #(batch,tokens,1) #(batch,tokens,1)

        start_logits=start_logits.squeeze(-1)
        end_logits=end_logits.squeeze(-1)
        #(batch,tokens) #(batch,tokens)
        
        return start_logits,end_logits
