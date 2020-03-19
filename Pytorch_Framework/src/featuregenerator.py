import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import pandas as pd


class Featureengineering():

    def __init__(self,df,cyc_feats,bin_feats,ord_feats,nom_feats):
        self.df=df
        self.cyclical_feats=cyc_feats
        self.bin_feat=bin_feats
        self.ord_feat=ord_feats
        self.nom_feat=nom_feats
        self.feats=bin_feats+ord_feats+nom_feats

    def cyclical_feat(self):
        '''
        To break the large diff before and after the split(12am and 1 am / 12 and 1 month)
        '''
        self.df['month_sin'] = np.sin((self.df[self.cyclical_feats[0]] - 1) * (2.0 * np.pi / 12))
        self.df['month_cos'] = np.cos((self.df[self.cyclical_feats[0]] - 1) * (2.0 * np.pi / 12))

        self.df['day_sin'] = np.sin((self.df[self.cyclical_feats[1]] - 1) * (2.0 * np.pi / 7))
        self.df['day_cos'] = np.cos((self.df[self.cyclical_feats[1]] - 1) * (2.0 * np.pi / 7))

        #self.df.drop(self.cyclical_feats,axis=1,inplace=True)

        self.cyclical_feats=['month_sin','month_cos','day_sin','day_cos']


    def label_encode(self):

        features = [x for x in self.df.columns 
            if x not in ['id', 'target'] + self.cyclical_feats]

        for feat in tqdm(features):
            lbl_enc = preprocessing.LabelEncoder()
    
            self.df[feat] = lbl_enc.fit_transform(self.df[feat]. \
                                         fillna('-1'). \
                                         astype(str).values)

        self.df['target'] = self.df['target'].fillna(-1)
        self.df[self.cyclical_feats] = self.df[self.cyclical_feats].fillna(-2)

    
    def ohe(self):
        self.df = pd.get_dummies(self.df,
                        columns=self.nom_feat[:5]+['month','day'],
                        sparse=True,
                        dtype=np.int8)

    def misc(self):
        self.df['ord_5_1'] = self.df['ord_5'].str[0]
        self.df['ord_5_2'] = self.df['ord_5'].str[1]
        self.df['nan_features'] = self.df.isna().sum(axis=1)
        
        self.df = self.df.drop('ord_5', axis=1)

    def engineer(self):
        self.misc()
        self.cyclical_feat()
        self.label_encode()
        self.ohe()
        return self.df