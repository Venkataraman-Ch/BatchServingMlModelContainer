#!/usr/bin/python3
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from joblib import dump
from sklearn import preprocessing

def train():
    MODEL_PATH_LDA  = '.\\modelfiles\\lda.joblib'
    MODEL_PATH_NN   = '.\\modelfiles\\nn.joblib'
    MODEL_PATH_DT   = '.\\modelfiles\\dt.joblib'
    
    # Load, read and normalize training data
    training = "./train.csv"
    data_train = pd.read_csv(training)
        
    y_train = data_train['# Letter'].values
    X_train = data_train.drop(data_train.loc[:, 'Line':'# Letter'].columns, axis=1)

    print("Shape of the training data")
    print(X_train.shape)
    print(y_train.shape)
        
    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')
    
    # Train Linear Discriminant Analysis (LDA)
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)
    dump(clf_lda, MODEL_PATH_LDA)
    
    # Train Neural Network (NN)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    clf_NN = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(500,), 
                           random_state=0, max_iter=1000)
    clf_NN.fit(X_train, y_train)
    dump(clf_NN, MODEL_PATH_NN)
    
    # Train Decision Tree Classifier (DT)
    clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt.fit(X_train, y_train)
    dump(clf_dt, MODEL_PATH_DT)

if __name__ == '__main__':
    train()
