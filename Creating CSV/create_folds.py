import pandas as pd
import numpy as np
import os
from sklearn import model_selection

if __name__ == '__main__':
    df = pd.read_csv('Input/train.csv')
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.label.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold_, (train_idx, valid_idx) in enumerate(kf.split(X=df,y=y)):
        print(fold_)
        df.loc[valid_idx,'kfold'] = fold_
    df.to_csv('Input/train_folds.csv',index=False)
