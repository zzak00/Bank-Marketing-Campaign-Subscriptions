import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


#extract categorical and numerical features
def extract_features_type(df):
    cat_col=[col for col in df.columns if df[col].dtype=='O']
    num_col=[col for col in df.columns if df[col].dtype!='O']
    return cat_col,num_col


#one hot encoding categorical variables
def one_hot_encoding(df,cat_col):
    dict={}
    for col in cat_col:
        if len(df[col].unique())>2:
            dict[col]=pd.get_dummies(df[col],drop_first=True)
    for key in dict.keys():
        df.drop(key,axis=1,inplace=True)
        df=pd.concat([df,dict[key]],axis=1)
    return df


#replace a list of strings with NAN
def replaceWNAN(df,list):
    for l in list:
        df=df.replace(l, np.nan)
    return df

#KNN IMPUTER
def knn_imput(data,k_neighbors):
    imputer = KNNImputer(n_neighbors=k_neighbors)
    data_imputer = pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
    return data_imputer


