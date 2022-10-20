

    
random.seed(12345)


ukp_gr = baps.merge(X_corr,right_index=True,left_index=True)
ukp_gr = ukp_gr.sort_values(by=["level 1"])
group = ukp_gr["level 1"]
X_corr = ukp_gr.drop(["level 1","level 2"], axis=1)
    
### Nested Cross Validation and Recursive Feature Elimination ###
#################################################################

for n in [100,50,30,10]:
    trainlist = {}
    auroc_te = {}
    auroc_tr = {}
    ClasRep =  pd.DataFrame()
    TPredictions = {}
    tPredictions = {}
    inner_auc = {}
    Features = list()
    i=0
    gss = GroupShuffleSplit(n_splits=16, test_size=.15, random_state=12345)
    for train_index, test_index in gss.split(X_corr,y,group):
        X_train, X_test = X_corr.iloc[train_index], X_corr.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_test.shape)
        i+=1
        trainlist[i] = list(X_train.index)

        ## CV inner
        features = list()  
        rf_te_roc_auc = list()
        rf_tr_roc_auc = list()
        acc = list()
        inner_score=list()

        ## inner split 

        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=12345)
        rf = RandomForestClassifier(random_state=1)
    ## define search space ##
        space = dict()
        space['n_estimators'] = [50, 100, 250,300] ## tree number space 
        space['max_depth'] = [4, 6, 8,10] ## tree depth space 
        search = GridSearchCV(rf, space, scoring='roc_auc', cv=sss, refit=True) 
        search.fit(X_train,y_train) 
        rf_cv = search.best_estimator_
        print(search.best_score_)
        inner_auc[i]=search.best_score_
        n_features_to_select = n  # Minimum number of features to consider
        rfe = RFE(estimator=rf_cv,n_features_to_select=n_features_to_select,step=10) ## Recursive feature elimination
        rfe.fit(X_train, y_train)


        ## Extract selected features ##

        f = rfe.get_support(1)
        X_tr_rfe = X_train[X_train.columns[f]] 
        X_te_rfe = X_test[X_train.columns[f]] 

        ## fit RF on training with selected features ##
        rf_rfe = rf_cv.fit(X_tr_rfe,y_train)
        yhat = rf_rfe.predict(X_te_rfe)
        yThat = rf_rfe.predict(X_tr_rfe)
        Features.append(list(X_te_rfe.columns))
        
        ## evaluate the model ##

        acc = accuracy_score(y_test, yhat)
        print("Accuracy: ",acc)
        ## AUROC ##
        rf_roc_auc = roc_auc_score(y_test,yhat)
        print("TestAUC: ",rf_roc_auc)
        auroc_te[i]= (rf_roc_auc) 
        rf_roc_auc = roc_auc_score(y_train,yThat)
        print("TrainAUC: ",rf_roc_auc)
        auroc_tr[i]= (rf_roc_auc) 
        ## Save Test Predictions
        data = {'Index': (y_test.index),
                'AET':y_test,
                    'Predictions':yhat, 
                   "Split":"Test",
                   "Iteration":i}
        Ptest = pd.DataFrame(data)
        tPredictions[i] = Ptest
        
        ## Save Train Predictions
        data = {'Index': (y_train.index),
                    'AET':y_train,
                        'Predictions':yThat, 
                       "Split":"Train",
                       "Iteration":i}
        Ptrain = pd.DataFrame(data)
        TPredictions[i] = Ptrain

        ## Scores ##
        cr = (classification_report(y_test, yhat,output_dict=True))
        cr = pd.DataFrame.from_dict(cr).T
        ClasRep = pd.concat([ClasRep,cr])
    
    path="OutputNewNCV/OuterGroupRFECV/RFE_CORR2/nselect"+str(n)+"/"
    ## Save AUC ##
    AUROCt = pd.DataFrame.from_dict(auroc_te, orient='index')
    AUROCt.to_csv(path+"testAUC.csv")
    AUROCT = pd.DataFrame.from_dict(auroc_tr, orient='index')
    AUROCT.to_csv(path+"trainAUC.csv")
        ## Figure
    AUROCt["Split"] = "Test"
    AUROCT["Split"] = "Train"
    auc = pd.concat([AUROCt,AUROCT])

    plt.rcParams["figure.figsize"] = (4,4)
    sns.set_style("whitegrid")
    fig=sns.boxplot(data=auc,x='Split',y=0,palette="Set3",width=0.4)
    plt.xlabel('')
    plt.ylabel('AUC')
    plt.savefig(path+"TrTeAUC.png",dpi=450)
    
    
    ### Select Features ###
    pd.DataFrame(Features).to_csv(path+"SelectedFeatures.csv")
    import itertools
    merged = list(itertools.chain(*Features))
    selected_uq_features = (set(merged))
    pd.DataFrame(selected_uq_features).to_csv(path+"SelectedUqFeatures.csv")

    #### Select features from auc > 0.7
    
    auc07 = list(auc[auc[0]>0.7].index)
    auc07 = [x - 1 for x in auc07]
    feats = pd.DataFrame(Features)
    feats_auc=pd.DataFrame()
    for n in auc07:
        a = (feats[feats.index==n])
        feats_auc = pd.concat([feats_auc,a])
    feats_auc.to_csv(path+"SelectedFeatures_auc07.csv")
    feats_auc = feats_auc.values.tolist()
    merged = list(itertools.chain(*feats_auc))
    selected_uq_features_auc07 = (set(merged))
    pd.DataFrame(selected_uq_features_auc07).to_csv(path+"SelectedUqFeatures_auc07.csv")


    ## Save Outer Classification Report ##
    ClasRep.to_csv(path+"ClassReport.csv")
    
   
### Cross Validation with Selected Feature Combinations ###
###########################################################

for i in ["100","50","30","10"]:
    selected_uq_features = list(pd.read_csv("OutputNCVFinal/RFE_CORR/nselect"+str(i)+"/SelectedUqFeatures_auc07.csv")["0"])
    ### Selected Features from NCV
    X_sel = X_corr[selected_uq_features]
    Xval_sel = X_val[selected_uq_features]

    ClasRep =  pd.DataFrame()

    ### Fit on whole training data ## 
    ### Test on validation/hold out data set ##


    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.15)#, random_state=12345)
    rf = RandomForestClassifier(random_state=12345)
    #         # define search space
    space = dict()
    space['n_estimators'] = [50, 100,200, 250,300]
    space['max_depth'] = [4, 6, 8,10]
    search = GridSearchCV(rf, space, scoring='roc_auc', cv=sss, refit=True)
    search.fit(X_sel,y)
    cv_results = search.cv_results_
    rf_cv = search.best_estimator_
    yhat = rf_cv.predict(Xval_sel)
    probas_ = rf_cv.predict_proba(Xval_sel)
    yThat = rf_cv.predict(X_sel)

    ## Features Importance ##
    features = X_sel.columns
    # Fmodels[i] = rf_rfe
    importances = rf_cv.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_imp_RF = (pd.DataFrame(np.flipud([features[z] for z in indices])))
    feat_imp_RF['Importance'] = importances

        ## Feature Permutation 
    resultp = permutation_importance(rf_cv, Xval_sel, y_val, n_repeats=500, n_jobs=-1)
    perm_sorted_idx = resultp.importances_mean.argsort()
    permutations = pd.DataFrame({'Feature':Xval_sel.columns.values[perm_sorted_idx][::-1],'Mean':resultp.importances_mean[perm_sorted_idx][::-1],
                          'std':resultp.importances_std[perm_sorted_idx][::-1]})
    PermutResults = permutations

    ## evaluate the model ##
    print("Testing on validation set: ")
    acc = accuracy_score(y_val, yhat)
    print(acc)

    ## AUROC test ##
    rf_roc_auc = roc_auc_score(y_val,yhat)

    print("Test AUC"+str(rf_roc_auc))
    auroc_te = (rf_roc_auc) 

    ## AUROC train ##
    rf_roc_auc = roc_auc_score(y,yThat)
    print("Train AUC"+str(rf_roc_auc))
    auroc_tr = (rf_roc_auc) 

    ## Save Test Predictions
    data = {'Index': (y_val.index),
                'AET':y_val,
                    'Predictions':yhat, 
                   "Split":"Test",
                   "Iteration":i}
    Ptest = pd.DataFrame(data)
    tPredictions = Ptest
        ## Save Train Predictions
    data = {'Index': (y.index),
                    'AET':y,
                        'Predictions':yThat, 
                       "Split":"Train",
                       "Iteration":i}
    Ptrain = pd.DataFrame(data)
    TPredictions = Ptrain
    
    ## Scores ##
    cr = (classification_report(y_val, yhat,output_dict=True))
    cr = pd.DataFrame.from_dict(cr).T
    ClasRep = pd.concat([ClasRep,cr])
    
    ## Save Feature Importance
    df = pd.DataFrame.from_dict(feat_imp_RF).sort_values(by=['Importance'],ascending=False)
    
## Feature selection - Importance based - ##
############################################

for i in ["100","50","30","10"]:
    
    df = pd.read_csv("OutputNCVFinal/RFE_CORR/nselect"+str(i)+"/Fall/FeatureImportance.csv")
    df = df[df['Importance'].cumsum() < 0.5]

    f = (round(df.shape[0]))
    print(f)
    X_seln = X_corr[list(df.iloc[0:f,]["0"])]
    Xval_seln = X_val[list(df.iloc[0:f,]["0"])]

    random.seed(12345)

    ClasRep =  pd.DataFrame()

    ### Fit on whole training data ## 
    ### Test on validation/hold out data set ##


    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=12345)
    rf = RandomForestClassifier(random_state=1)
    #         # define search space
    space = dict()
    space['n_estimators'] = [50, 100,200, 250,300]
    space['max_depth'] = [4, 6, 8,10]
    search = GridSearchCV(rf, space, scoring='roc_auc', cv=sss, refit=True)
    search.fit(X_seln,y)
    cv_results = search.cv_results_
    rf_cv = search.best_estimator_
    yhat = rf_cv.predict(Xval_seln)
    probas_ = rf_cv.predict_proba(Xval_seln)
    yThat = rf_cv.predict(X_seln)

    ## Features Importance ##
    
    features = X_seln.columns
    importances = rf_cv.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_imp_RF = (pd.DataFrame(np.flipud([features[z] for z in indices])))
    feat_imp_RF['Importance'] = importances

        ## Feature Permutation on Test
      
    resultp = permutation_importance(rf_cv, Xval_seln, y_val, n_repeats=100, n_jobs=-1)
    perm_sorted_idx = resultp.importances_mean.argsort()
    permutations = pd.DataFrame({'Feature':Xval_seln.columns.values[perm_sorted_idx][::-1],'Mean':resultp.importances_mean[perm_sorted_idx][::-1],
                          'std':resultp.importances_std[perm_sorted_idx][::-1]})
    PermutResultst = permutations
   
  ## plot permutation importance ##
    
    tree_importance_sorted_idx = np.argsort(rf_cv.feature_importances_)
    tree_indices = np.arange(0, len(rf_cv.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices, rf_cv.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(Xval_seln.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(rf_cv.feature_importances_)))
    ax2.boxplot(
        resultp.importances[perm_sorted_idx].T,
        vert=False,
        labels=Xval_seln.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()
    
            ## Feature Permutation on Train
    resultp = permutation_importance(rf_cv, X_seln, y, n_repeats=100, n_jobs=-1)
    perm_sorted_idx = resultp.importances_mean.argsort()
    permutations = pd.DataFrame({'Feature':X_seln.columns.values[perm_sorted_idx][::-1],'Mean':resultp.importances_mean[perm_sorted_idx][::-1],
                          'std':resultp.importances_std[perm_sorted_idx][::-1]})
    PermutResultsT = permutations
        ##plot
    tree_importance_sorted_idx = np.argsort(rf_cv.feature_importances_)
    tree_indices = np.arange(0, len(rf_cv.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices, rf_cv.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(X_seln.columns[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(rf_cv.feature_importances_)))
    ax2.boxplot(
        resultp.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_seln.columns[perm_sorted_idx],
    )
    fig.tight_layout()
    plt.show()
    
    
    ## evaluate the model ##

    acc = accuracy_score(y_val, yhat)
    print(acc)

    ## AUROC test ##
    rf_roc_auc = roc_auc_score(y_val,yhat)

    print("Test AUC"+str(rf_roc_auc))
    auroc_te = (rf_roc_auc) 

    ## AUROC train ##
    rf_roc_auc = roc_auc_score(y,yThat)
    print("Train AUC"+str(rf_roc_auc))
    auroc_tr = (rf_roc_auc) 

    ## Save Test Predictions
    data = {'Index': (y_val.index),
                'AET':y_val,
                    'Predictions':yhat, 
                   "Split":"Test",
                   "Iteration":i}
    Ptest = pd.DataFrame(data)
    tPredictions = Ptest
        ## Save Train Predictions
    data = {'Index': (y.index),
                    'AET':y,
                        'Predictions':yThat, 
                       "Split":"Train",
                       "Iteration":i}
    Ptrain = pd.DataFrame(data)
    TPredictions = Ptrain

    ## Scores ##
    cr = (classification_report(y_val, yhat,output_dict=True))
    cr = pd.DataFrame.from_dict(cr).T
    ClasRep = pd.concat([ClasRep,cr])
    ## Save the model ##
    
    joblib.dump(rf_cv, "OutputNCVFinal/RFE_CORR/nselect"+str(i)+"/random_forest.joblib")

