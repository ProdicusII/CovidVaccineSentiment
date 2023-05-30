#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:51:34 2022

@author: nidzoni
"""
import numpy as np
import pickle
from dataset import TweetsDataset
from transformers import AutoTokenizer
modelname='classla/bcms-bertic'  ## u ovom slucaju moze ovako, ali moze i './BERTICsentiments'
tokenizer = AutoTokenizer.from_pretrained(modelname)  ## loading the tokenizer of that model


modelname='./BERTICsentiments'
from transformers import ElectraForSequenceClassification #AutoModelForPreTraining  
model = ElectraForSequenceClassification.from_pretrained(modelname, num_labels=3)
#val_dataset=torch.load('./val_dataset.pt')

filename = './test_sentiments.pickle'
infile = open(filename,'rb')
bb = pickle.load(infile)
dataset=bb[0]
ids=bb[1]
infile.close()


from transformers import Trainer
evaler = Trainer(
    model=model
)
pred=evaler.predict(dataset)


from sklearn.metrics import confusion_matrix, recall_score, f1_score
matrix=confusion_matrix( pred[1], pred[0].argmax(axis=1))
recall=recall_score( pred[1], pred[0].argmax(axis=1))                                                               
f1=f1_score( pred[1], pred[0].argmax(axis=1))                                                                           

predictions=[]
predictions.append(pred[0].argmax(axis=1))
predictions.append(pred[1])
predictions=np.array(predictions).T

confuses=      [  [dict(),dict(),dict()] , [dict(),dict(),dict()] , [dict(),dict(),dict()]  ]

encs=dataset.encodings['input_ids']
encs=np.array(encs)

for vesnaclass in range(3):
    for algclass in range(3):
        mask=(predictions[:,0]==algclass) & (predictions[:,1]==vesnaclass)
        selencs=list(encs[mask])
        dic=dict()
        dic['ids']=list(ids[mask])
        dic['model_outputs']=pred[0][mask]
        lis=[]
        for seq in selencs:
            lis.append(tokenizer.decode(seq))
        dic['tweets']=lis
        confuses[vesnaclass][algclass]=dic
        confuses[vesnaclass][algclass]['joined']='\n\n'.join(confuses[vesnaclass][algclass]['tweets'])

filename = './confuses_sentiments.pickle'
infile = open(filename,'wb')
pickle.dump(confuses,infile)
infile.close()

#tabledic={'vesna':[],'algo':[],'text':[],'id':[]}
import pandas as pd

conf=pd.DataFrame()
for vesnaclass in range(3):
    for algclass in range(3):
        if vesnaclass != algclass:
            panda=pd.DataFrame()
            duz=len(confuses[vesnaclass][algclass]['ids'])
            panda['tweets']=confuses[vesnaclass][algclass]['tweets']
            panda['ids']=confuses[vesnaclass][algclass]['ids']
            panda['vesna']=np.ones(duz)*vesnaclass
            panda['algorithm']=np.ones(duz)*algclass
            conf=pd.concat([conf,panda])
            conf.reset_index(drop=True,inplace=True)
conf['nik']=8
conf.to_excel('confusion_sentiments.xlsx')