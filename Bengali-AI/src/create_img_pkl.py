import glob
import joblib
from tqdm import tqdm
import pandas as pd


if __name__=="__main__":
    files=glob.glob('../inputs/bengaliai-cv19/*.parquet')

    for f in files:
        df=pd.read_parquet(f)
        image_ids=df['image_id'].values
        df=df.drop('image_id',axis=1)
        img_arrays=df.values

        for i,img_id in tqdm(enumerate(image_ids),total=len(image_ids)):
            joblib.dump(img_arrays[i,:],f'../inputs/image_pickles/{img_id}.pkl')
