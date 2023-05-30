#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:55:47 2022

@author: nidzoni
"""

import pandas as pd
import numpy as np

def preproc(texts):
    import re
    texts=texts.apply(lambda xx:re.sub('http[\w://\.]+','',xx))   #sklanja linkove
    texts=texts.apply(lambda xx:re.sub('@\w+\s','',xx))           # sklanja reference na usere
    texts=texts.apply(lambda xx:re.sub('[\U00002500-\U000EFFFF]','',xx)) #sklanja hijeroglife
    texts=texts.apply(lambda xx:re.sub('#vojvodina','',xx))  #sklanja ova dva hastaga vojvodina i covid19 koji se masovno pojavljuju
    texts=texts.apply(lambda xx:re.sub('#COVID19','',xx))    #sklanja hijeroglife
    texts=texts.apply(lambda xx:re.sub(r'\\n',' ',xx))    #zamenjuje new line sa space om
    pattern=re.compile(r'(#)(\w+\b)')
    texts=texts.apply(lambda xx:pattern.sub((r'\2'.capitalize()),xx))  # zamenjuje hashtag i Nece da kapitalizuje :(
    return texts

# original0=pd.read_csv('stari_obelezeni_tvitovi_dijakriticki.csv')
# original0.drop(11686,axis=0,inplace=True)
# original0.drop(11433,axis=0,inplace=True)

# original0.rename(columns={"text": "text_original", "textLatinicaDijakriticki": "text"}, inplace=True)
##############################################################################################
# original=pd.read_excel('tvitovi.xlsx')
# original.drop(11433,axis=0,inplace=True)
# original['id']=original['id'].apply(lambda xx: str(np.int(xx)))

# original0['id']=original['id']

# original0.to_excel('anotirani_tvitovi.xlsx')


original0=pd.read_excel('anotirani_tvitovi.xlsx',dtype={'ids':str})


tweets=original0[['text','Vesna','id']]
tweets=tweets[tweets['Vesna'].notna()]
tweets[tweets['Vesna'].where(tweets['Vesna']=='?').isna()]


for index, item in tweets.iterrows():
    # print(item)
    if type(item['Vesna'])==str :
        tweets.drop(index=index,inplace=True)

        
tvitovi=tweets['text']
tweets.loc[:,'text']=preproc(tvitovi)

relevants=tweets.copy(deep=True)
relevants['Vesna']=relevants['Vesna'].replace([0,2,4],1)
relevants['Vesna']=relevants['Vesna'].replace([7,8],0)



tweets=tweets[(tweets['Vesna']>=0) & (tweets['Vesna']<=4)]
# tweets['Vesna']=tweets['Vesna'].replace(2,1)
# tweets['Vesna']=tweets['Vesna'].replace(4,2)


tags1=tweets[['Vesna']].to_numpy() /2  ###   lists of tags that we predict and train. U ovom slucaju UD tags
tags1=tags1.astype(int)
tags1=list(tags1)
tweets.loc[:,['Vesna']]=tags1

tags1=relevants[['Vesna']].to_numpy()  ###   lists of tags that we predict and train. U ovom slucaju UD tags
tags1=tags1.astype(int)
tags1=list(tags1)
relevants.loc[:,['Vesna']]=tags1


tweets.reset_index(drop=True).to_pickle('./sentiments/sentiments.pickle')
relevants.reset_index(drop=True).to_pickle('./relevants/relevants.pickle')
