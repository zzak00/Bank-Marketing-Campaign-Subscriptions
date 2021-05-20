from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image 
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm

def svm(X,y):
    clf = SVC()
    clf.fit(X, y)
    return clf

def knn(x,y,n):
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(x,y)
    return knn
def DT(x,y):
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    return dt
def MLP(x,y):
    clf = MLPClassifier(solver='adam',hidden_layer_sizes=(8,7,7),activation='relu',batch_size=32, max_iter=300).fit(x, y.values.ravel())
    return clf
def report(model,x_test,y_test):
    print(classification_report(y_test,model.predict(x_test)))

def LR(x,y):
    logisticRegr = LogisticRegression(C=10e10)
    logisticRegr.fit(x, y)
    return logisticRegr

#models is an array like that 
'''
models = []
models.append(('MLP', MLPClassifier(solver='adam',hidden_layer_sizes=(8,7,7),activation='relu',batch_size=32, max_iter=300)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('decision tree', DecisionTreeClassifier()))
models.append(('logistic',LogisticRegression(C=10e10)))
# evaluate each model in turn'''
def COMPARE(models,x_smote,y_smote):
    seed = 7

    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
        cv_results = model_selection.cross_val_score(model, x_smote, y_smote.values.ravel(), cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()