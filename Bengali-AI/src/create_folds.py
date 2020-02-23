import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../inputs/bengaliai-cv19/train.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    X=df['image_id'].values
    y=df[["grapheme_root","vowel_diacritic","consonant_diacritic"]].values

    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=False, random_state=42)


    for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    print("counts",df['kfold'].value_counts())
    df.to_csv("../inputs/train_folds.csv", index=False)