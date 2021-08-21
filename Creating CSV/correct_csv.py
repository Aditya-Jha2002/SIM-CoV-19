import pandas as pd

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    df = pd.read_csv('Input/train_study_level.csv')
    df['label'] = -1
    df['class'] = -1
    def label_to_class(x):
        if x == 0:
            return 'Negative for Pneumonia'
        elif x == 1:
            return 'Typical Appearance'
        elif x == 2:
            return 'Indeterminate Appearance'
        else:
            return 'Atypical Apperance'
    df['label'] = np.where(np.array(df[['Negative for Pneumonia','Typical Appearance','Indeterminate Appearance','Atypical Appearance']]==1))[1]
    df['class'] = df['label'].apply(label_to_class)
    df.drop(['Negative for Pneumonia', 'Typical Appearance',
        'Indeterminate Appearance', 'Atypical Appearance'],axis=1,inplace=True)
    df.to_csv('Input/train.csv',index=False)
