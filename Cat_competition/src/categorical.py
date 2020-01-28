from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd

'''
-Binarization
-One hot encoding
-Label encoding

'''

class CategoricalFeatures:
    '''
    df- Pandas dataframe
    cat_features - Categorical features
    encoding_type - type of encoding - label,ohe,binary
    handle_na - True/False
    '''

    def __init__(self,df,cat_features,encoding_type,handle_na=False):
        self.df=df
        self.cat_features=cat_features
        self.encoding_type=encoding_type
        self.handle_na=handle_na
        self.label_encoders=dict()
        self.binary_encoders=dict()
        self.ohe=None

        for c in self.cat_features:
            self.df.loc[:,c]=self.df.loc[:,c].astype(str).fillna('-999999')

        self.out_df=self.df.copy(deep=True)
        
    def _label_binarizer(self):

        for c in self.cat_features:
            lbl=preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val=lbl.transform(self.df[c].values)    #array [[0,0,1],[1,0,0]]
            self.out_df=self.out_df.drop(c,axis=1)

            for j in range(val.shape[1]):
                new_col=c+f'bin_{j}'
                self.out_df[new_col]=val[:,j]
            self.binary_encoders[c]=lbl

        return self.out_df

    
    def _label_encoding(self):

        for c in self.cat_features:
            lbl=preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.out_df.loc[:,c]=lbl.transform(self.df[c].values)
            self.label_encoders[c]=lbl
    
        return self.out_df


    def _ohe(self):
        self.ohe=preprocessing.OneHotEncoder()
        self.ohe.fit(self.df[self.cat_features].values)

        return self.ohe.transform(self.df[self.cat_features].values)



    def fit_transform(self):

        if self.encoding_type=='label':
            return self._label_encoding()

        elif self.encoding_type=='binary':
            return self._label_binarizer()

        elif self.encoding_type=='onehot':
            return self._ohe()

        else:
            raise Exception("Encoding type not understood")


    def transform(self,dataframe):

        if self.handle_na:
            for c in self.cat_features:
                dataframe.loc[:,c]=dataframe.loc[:,c].astype(str).fillna('-999999')

        if self.encoding_type=='label':
            for c,lbl in self.label_encoders.items():
                dataframe.loc[:,c]=lbl.transform(dataframe[c].values)

            return dataframe

        elif self.encoding_type=='binary':
            for c,lbl in self.binary_encoders.items():
                val=lbl.transform(dataframe[c].values)
                dataframe=dataframe.drop(c,axis=1)

                for j in range(val.shape[1]):
                    new_col=c+f'bin_{j}'
                    dataframe[new_col]=val[:,j]

            return dataframe
        
        elif self.encoding_type=='onehot':
            return self.ohe(dataframe[self.cat_features].values)

        else:
            raise Exception("Encoding type not understood")    
        



if __name__== "__main__":
        df=pd.read_csv('../inputs/trainv2.csv')
        df_test=pd.read_csv('../inputs/testv2.csv')
        sample_submission=pd.read_csv('../inputs/sample_submissionv2.csv')

        df_test['target']=-1
        full_data=pd.concat([df,df_test])

        train_len=len(df)

        #train_idx=df['id'].values
        #test_idx=df_test['id'].values

        cols=[c for c in full_data.columns if c not in ["id","target"]]

        print("columns",cols)

        cats=CategoricalFeatures(full_data,cat_features=cols,encoding_type='onehot',handle_na=True)

        full_transform=cats.fit_transform()

        X = full_transform[:train_len, :]
        X_test = full_transform[train_len:, :]
        
        # train_df=full_transform[full_transform['id'].isin(train_idx)].reset_index(drop=True)
        # test_df=full_transform[full_transform['id'].isin(test_idx)].reset_index(drop=True)

        clf=LogisticRegression()
        clf.fit(X,df.target.values)

        preds=clf.predict_proba(X_test)[:,1]

        sample_submission.loc[:,"target"]=preds
        sample_submission.to_csv('../inputs/submission.csv',index=False)