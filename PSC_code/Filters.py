# IMPORT #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2




### FILTERS ###
###############

## feature selection using chi2 on training ## 

def chi2_test(X_train,y_train,X_test):
    bestfeatures = SelectKBest(score_func=chi2, k="all")
    fit = bestfeatures.fit(X_train,y_train)
    
    #create df for scores
    dfscores = pd.DataFrame(fit.scores_)
    dfpvalues = pd.DataFrame(fit.pvalues_)
    
    #create df for column names
    dfcolumns = pd.DataFrame(X_train.columns)
    
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
    
    #naming the dataframe columns
    featureScores.columns = ['Selected_columns','Score_chi2','p_val']
    
    ### subset the original features set with the features p<0.01
    X_new=bestfeatures.fit_transform(X_train, y_train)
    featureScores001 = featureScores[featureScores.p_val < 0.01]
    X_tr_chi2 = X_train[featureScores001.Selected_columns]
    X_te_chi2 = X_test[featureScores001.Selected_columns]
    print("Seleced Chi Features",X_tr_chi2.shape)
    return X_tr_chi2,X_te_chi2,featureScores

## REMOVE correlated variables ###

def correlation(X_train,X_test,n):    
    # Create correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.7
    to_drop = [column for column in upper.columns if any(upper[column] > n)]
    # Drop features 
    X_tr_corr = X_train.drop(to_drop, axis=1)#, inplace=True)
    X_te_corr = X_test.drop(to_drop, axis=1)
    return X_tr_corr,X_te_corr,corr_matrix


X_chi2,X_val_chi2,featureScores = chi2_test(X,y,X_val)
X_corr,X_val_corr,CorMatrix = correlation(X_chi2,X_val_chi2,0.7)
