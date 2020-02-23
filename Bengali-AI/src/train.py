import os
import ast
import torch
import torch.nn as nn
from modeldispatcher import MODEL_DISPATCHER
from dataset import BengaliAiDataset
from tqdm import tqdm

DEVICE='cuda'
TRAINING_FOLDS_CSV=os.environ.get('TRAINING_FOLDS_CSV')
IMG_HEIGHT=int(os.environ.get('IMG_HEIGHT'))
IMG_WIDTH=int(os.environ.get('IMG_WIDTH'))
EPOCHS=int(os.environ.get('EPOCHS'))

TRAIN_BATCHSIZE=os.environ.get('TRAIN_BATCHSIZE')
TEST_BATCHSIZE=os.environ.get('TEST_BATCHSIZE')

MODEL_MEAN=ast.literal_eval(os.environ.get('TRAIN_BATCHSIZE'))
MODEL_STD=ast.literal_eval(os.environ.get('TEST_BATCHSIZE'))


TRAINING_FOLDS=ast.literal_eval(os.environ.get('TRAINING_FOLDS'))
VALIDATION_FOLDS=ast.literal_eval(os.environ.get('VALIDATION_FOLDS'))
BASE_MODEL=os.environ.get('BASE_MODEL')


def train(dataset,dataloader,model,optimizer):
    model.train()

    for bs,data in tqdm(enumerate(dataloader),total=int(len(dataset)/dataloader.batch_size)):
        image=data['image']
        grapheme_root=data['grapheme_root']
        vowel_diacritic=data['vowel_diacritic']
        consonant_diacritic=data['consonant_diacritic']

        image=image.to(DEVICE,dtype=torch.float)
        grapheme_root=grapheme_root.to(DEVICE,dtype=torch.long)
        vowel_diacritic=vowel_diacritic.to(DEVICE,dtype=torch.long)
        consonant_diacritic=consonant_diacritic.to(DEVICE,dtype=torch.long)

        optimizer.zero_grad()
        outputs=model(image)
        targets=(grapheme_root,vowel_diacritic,consonant_diacritic)

        loss=loss_func(outputs,targets)
        
        loss.backward()
        optimizer.step()

def loss_func(outputs,targets):
    01,02,03=outputs
    t1,t2,t3=targets

    l1=nn.CrossEntropyLoss()(01,t1)
    l2=nn.CrossEntropyLoss()(02,t2)
    l3=nn.CrossEntropyLoss()(03,t3)

    return (l1+l2+l3)/3

def evaluate(dataset,dataloader,model):
    model.eval()
    final_loss=0
    counter=0

    for bs,data in tqdm(enumerate(dataloader),total=int(len(dataset)/dataloader.batch_size)):
        image=data['image']
        grapheme_root=data['grapheme_root']
        vowel_diacritic=data['vowel_diacritic']
        consonant_diacritic=data['consonant_diacritic']

        image=image.to(DEVICE,dtype=torch.float)
        grapheme_root=grapheme_root.to(DEVICE,dtype=torch.long)
        vowel_diacritic=vowel_diacritic.to(DEVICE,dtype=torch.long)
        consonant_diacritic=consonant_diacritic.to(DEVICE,dtype=torch.long)

        outputs=model(image)
        targets=(grapheme_root,vowel_diacritic,consonant_diacritic)

        loss=loss_func(outputs,targets)

        final_loss+=loss

    return loss/counter

def main():

    model=MODEL_DISPATCHER[BASE_MODEL][pretrained=True]
    model.to(DEVICE)

    train_dataset=BengaliAiDataset(
        folds=TRAINING_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=TRAIN_BATCHSIZE,
                                            shuffle=True,
                                            num_workers=4)


    valid_dataset=BengaliAiDataset(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader=torch.utils.data.DataLoader(dataset=valid_dataset,
                                            batch_size=TEST_BATCHSIZE,
                                            shuffle=False,
                                            num_workers=4)


    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                mode='min',patience=5,verbose=True,   #mode - max if we want to increase let's say recall, if we want to decrease loss then it will be min
                                                factor=0.3)

    
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)

    
    for epoch in EPOCHS:
        train(train_dataset,train_loader,model,optimizer)
        val_score=evaluate(valid_dataset,valid_loader,model)
        scheduler.step(val_score)
        torch.save(model.state_dict(),f'{BASEMODEL}_fold_{VALIDATION_FOLDS[0]}.bin')




if __name__=="__main__":
    main()

