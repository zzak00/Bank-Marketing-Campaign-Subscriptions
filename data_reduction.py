from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import math

def ANOVA(df,X,y):
    fs = SelectKBest(score_func=f_classif, k=3)
    # learn relationship from training data
    X_train, X_test, y_train, y_test =train_test_split(X,y,train_size=0.3, random_state=0) 
    fs.fit(X_train, y_train)

    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)

    #order features
    fb={}
    for i in range(len(fs.scores_)):
        fb[df.columns[i]]=fs.scores_[i]
    sort_orders = sorted(fb.items(), key=lambda x: x[1], reverse=True)
    fb={}
    for i in range(len(fs.scores_)):
        fb[df.columns[i]]=fs.scores_[i]
    sort_orders = sorted(fb.items(), key=lambda x: x[1], reverse=True)
    subset=[]
    for i in range(0,27):
        a,b=sort_orders[i]
        if math.isnan(b)!=True:
            subset.append(a)
    return subset


def RF(X,y):
    X_train, X_test, y_train, y_test =train_test_split(X,y,train_size=0.3, random_state=0) 
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_train, y_train)
    return X.columns[(sel.get_support())]

def MI(X,y,i):
    imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
    imp.columns=['importance']
    imp.sort_values(by='importance',ascending=False)
    best_subset=imp.sort_values(by='importance',ascending=False)[:i].index
    return best_subset
    
