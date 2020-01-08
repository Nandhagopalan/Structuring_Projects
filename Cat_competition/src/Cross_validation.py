
"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""

import pandas as pd
from sklearn import model_selection


class Crossvalidation():

    def __init__(
        self,
        df,
        target_cols,
        problem_type='binary_classification',
        num_folds=5,
        shuffle,
        multilabel_delimiter=",",
        random_state=42
        ):

        self.df=df
        self.target=target_cols
        self.num_targets=len(target_cols)
        self.problem_type=problem_type
        self.num_folds=num_folds
        self.problem_type=problem_type
        self.num_folds=num_folds
        self.shuffle=shuffle
        self.random_state=random_state
        self.multilabel_delimiter = multilabel_delimiter
        self.df['kfold']=-1

        if self.shuffle:
            self.df=self.df.sample(frac=1).reset_index(drop=True)

        
    def split(self):

        if self.problem_type in ("binary_classification", "multiclass_classification"):

            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target[0]
            unique_values = self.df[target].nunique()
            
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                    self.df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            
            kf = model_selection.KFold(n_splits=self.num_folds)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df)):
                self.df.loc[val_idx, 'kfold'] = fold


        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.df) * holdout_percentage / 100)
            self.df.loc[:len(self.df) - num_holdout_samples, "kfold"] = 0
            self.df.loc[len(self.df) - num_holdout_samples:, "kfold"] = 1



        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

            targets = self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=targets)):
                self.df.loc[val_idx, 'kfold'] = fold


        return self.df



if __name__ == "__main__":
    df = pd.read_csv("../inputs/train_multilabel.csv")
    cv = CrossValidation(df, shuffle=True, target_cols=["attribute_ids"], 
                         problem_type="multilabel_classification", multilabel_delimiter=" ")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
