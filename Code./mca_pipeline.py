

# Create list of tuples with classifier label and classifier object
classifiers = {}
classifiers.update({"Random Forest": RandomForestClassifier()})

# Initiate parameter grid
parameters = {}

# Update dict with Random Forest Parameters
parameters.update({"Random Forest": { 
                                    "classifier__n_estimators": [50,100,200],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                    "classifier__n_jobs": [-1]
                                     }})


########################################################
########## No population structure control #############
########################################################


path="NoPopControl/"
SelFeatures = dict()
i=0
trainlist = dict()
RESULTS = dict()
AUC_SCORES=dict()
TPREDS = dict()
tPREDS = dict()
FeatImp = list()


###### Outer Fold Split X_train X_test #######
##############################################

out_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

for train_index, test_index in out_split.split(mca_aet.drop(["R2_AET"],axis=1),mca_aet["R2_AET"]):
    strat_train_set = mca_aet.iloc[train_index]
    strat_test_set = mca_aet.iloc[test_index]
            
    y_train = strat_train_set["R2_AET"]
    X_train = strat_train_set.drop(["R2_AET"],axis=1)
    y_test = strat_test_set["R2_AET"]
    X_test = strat_test_set.drop(["R2_AET"],axis=1)
    i+=1
    trainlist[i] = list(X_train.index)
                  
###############################################################################
#                     1. Tuning a classifier to use with RFECV                #
###############################################################################
    # Define classifier to use as the base of the recursive feature elimination algorithm
    selected_classifier = "Random Forest"
    classifier = classifiers[selected_classifier]

    # Tune classifier 

    # Define steps in pipeline
    steps = [("classifier", classifier)]

    # Initialize Pipeline object
    pipeline = Pipeline(steps = steps)

    # Define parameter grid
    param_grid = parameters[selected_classifier]

    # Initialize GridSearch object
    gscv = GridSearchCV(pipeline, param_grid, cv = 3,  n_jobs= -1, verbose = 1, scoring = "roc_auc")

    # Fit gscv
    print(f"Now tuning {selected_classifier}.")
    gscv.fit(X_train, np.ravel(y_train))  

    # Get best parameters and score
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    # Update classifier parameters
    tuned_params = {item[12:]: best_params[item] for item in best_params}
    classifier.set_params(**tuned_params)
    
    ###############################################################################
#                2. Feature Selection: Recursive Feature Selection           #
###############################################################################
# Select Features using RFECV
    class PipelineRFE(Pipeline):
        # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
        def fit(self, X, y=None, **fit_params):
            super(PipelineRFE, self).fit(X, y, **fit_params)
            self.feature_importances_ = self.steps[-1][-1].feature_importances_
            return self
    # Define pipeline for RFE
    steps = [ ("classifier", classifier)]
    pipe = PipelineRFE(steps = steps)

    # Initialize RFE object
    feature_selector = RFE(pipe, n_features_to_select = 10, step = 1, verbose = 1)

    # Fit RFE
    feature_selector.fit(X_train, np.ravel(y_train))

    # Get selected features labels
    feature_names = X_train.columns
    selected_features = feature_names[feature_selector.support_].tolist()
    SelFeatures[i]= selected_features
    print("RFE selected Features", selected_features)

    ###############################################################################
#                  3. Visualizing Selected Features Importance               #
###############################################################################
    # Get selected features data set
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Train classifier
    classifier.fit(X_train, np.ravel(y_train))

    # Get feature importance
    feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])
    feature_importance["Feature Importance"] = classifier.feature_importances_

    # Sort by feature importance
    feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)
    FeatImp.append(feature_importance)
    # Set graph style
    sns.set(font_scale = 1.75)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                   "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                   'ytick.color': '0.4'})

    # Set figure size and create barplot
    f, ax = plt.subplots(figsize=(12, 9))
    sns.barplot(x = "Feature Importance", y = "Feature Label",
                palette = reversed(sns.color_palette('YlOrRd', 15)),  data = feature_importance)

    # Generate a bolded horizontal line at y = 0
    ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

    # Turn frame off
    ax.set_frame_on(False)

    # Tight layout
    plt.tight_layout()

    # Save Figure
    plt.savefig(path+"feature_importance"+str(i)+".png", dpi = 1080)
    
    ###############################################################################
#                       4. Classifier Tuning and Evaluation                  #
###############################################################################
# Initialize dictionary to store results
    results = {}
    TPredictions = {}
    tPredictions = {}   


    predictions = classifier.predict(X_test)
    auc_te = metrics.roc_auc_score(y_test,predictions)
                ## save individual predictions
    data = {'Index': (y_test.index),
            'AET':y_test,
                'Predictions':predictions, 
               "Split":"Test",
               "Iteration":i}
    Ptest = pd.DataFrame(data)
    tPredictions[i] = Ptest

    tr_predictions = classifier.predict(X_train)
    auc_tr = metrics.roc_auc_score(y_train, tr_predictions)

    data = {'Index': (y_train.index),
            'AET':y_train,
                'Predictions':tr_predictions, 
               "Split":"Train",
               "Iteration":i}
    Ptrain = pd.DataFrame(data)
    TPredictions[i] = Ptrain
    # Save results
    result = {"Classifier": classifier,
                  "Best Parameters": best_params,
                  "Training AUC": auc_tr,
                  "Test AUC": auc_te}

    RESULTS[i]=result
    TPREDS[i]=(TPredictions)
    tPREDS[i]=(tPredictions)
    
    
#################################    
### CV with Selected Features ###
#################################

X = mca_aet[set_features]
y = mca_aet["R2_AET"]

scores_te = list()
scores_tr = list()

feat_imp = list()
ClasRepCV = pd.DataFrame()
feat_imp_RFCV = pd.DataFrame()
PermutResultsRFCV = pd.DataFrame()
tPredictions = {}
TPredictions = {}
i=0

out_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

for train_index, test_index in out_split.split(X,y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    i+=1
    
    model = RandomForestClassifier(bootstrap = True,
                                   oob_score= True,
                                  n_estimators=100,
                                  max_depth=4,
                                  class_weight='balanced',
                                  max_features='sqrt')
    model.fit(X_train,y_train)
    # evaluate model
    yhat = model.predict(X_test)
    tr_predictions = model.predict(X_train)
    
    data = {'Index': (y_test.index),
            'AET':y_test,
                'Predictions':yhat, 
               "Split":"Test",
               "Iteration":i}
    Ptest = pd.DataFrame(data)
    tPredictions[i] = Ptest

    data = {'Index': (y_train.index),
            'AET':y_train,
                'Predictions':tr_predictions, 
               "Split":"Train",
               "Iteration":i}
    Ptrain = pd.DataFrame(data)
    TPredictions[i] = Ptrain
        
    roc_auc = roc_auc_score(y_test, yhat)
    # store score
    scores_te.append(roc_auc)
    
    roc_auc_tr = roc_auc_score(y_train, tr_predictions)
    # store score
    scores_tr.append(roc_auc_tr)
    print('> ', roc_auc)
    cr = (classification_report(y_test, yhat,output_dict=True))
    cr = pd.DataFrame.from_dict(cr).T
    ClasRepCV = pd.concat([ClasRepCV,cr])
    features = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feats = pd.DataFrame(np.flipud([features[z] for z in indices]))
    feats["Importance"] = importances #= (pd.DataFrame(importances))
    feat_imp_RFCV = pd.concat([feat_imp_RFCV,feats.sort_values("Importance", ascending=False)])
    #             Feature Permutation 
    resultp = permutation_importance(model, X_train, y_train, n_repeats=10, n_jobs=-1)
    perm_sorted_idx = resultp.importances_mean.argsort()
    permutations = pd.DataFrame({'Feature':X_train.columns.values[perm_sorted_idx][::-1],'Mean':resultp.importances_mean[perm_sorted_idx][::-1],
                  'std':resultp.importances_std[perm_sorted_idx][::-1]})
    PermutResultsRFCV = pd.concat([PermutResultsRFCV,permutations])

# summarize model performance
mean_s, std_s = mean(scores_te), std(scores_te)
print('Mean: %.3f, Standard Deviation: %.3f' % (mean_s, std_s))



#######################################################################
################## WITH population structure control #################
#######################################################################



path="PopControl/"
SelFeatures = dict()
i=0
trainlist = dict()
RESULTS = dict()
AUC_SCORES=dict()
TPREDS = dict()
tPREDS = dict()
FeatImp = list()
X=mca_aet.drop(["R2_AET"],axis=1)
y=mca_aet["R2_AET"]
                  
###### Outer Fold Split X_train X_test #######
##############################################

gss = GroupShuffleSplit(n_splits=16, test_size=.15, random_state=12345)
    #     group_kfold = GroupKFold(n_splits=10,shuffle=False)
for train_index, test_index in gss.split(X,y,group-1):
    #         print("TRAIN:", train_index, "TEST:", test_inde)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_test.shape)
    i+=1
    trainlist[i] = list(X_train.index)
                  
###############################################################################
#                     1. Tuning a classifier to use with RFECV                #
###############################################################################
    # Define classifier to use as the base of the recursive feature elimination algorithm
    selected_classifier = "Random Forest"
    classifier = classifiers[selected_classifier]

    # Tune classifier 

    # Define steps in pipeline
    steps = [("classifier", classifier)]

    # Initialize Pipeline object
    pipeline = Pipeline(steps = steps)

    # Define parameter grid
    param_grid = parameters[selected_classifier]

    # Initialize GridSearch object
    gscv = GridSearchCV(pipeline, param_grid, cv = 3,  n_jobs= -1, verbose = 1, scoring = "roc_auc")

    # Fit gscv
    print(f"Now tuning {selected_classifier}.")
    gscv.fit(X_train, np.ravel(y_train))  

    # Get best parameters and score
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    # Update classifier parameters
    tuned_params = {item[12:]: best_params[item] for item in best_params}
    classifier.set_params(**tuned_params)
    
    ###############################################################################
#                2. Feature Selection: Recursive Feature Selection           #
###############################################################################
# Select Features using RFECV
    class PipelineRFE(Pipeline):
        # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
        def fit(self, X, y=None, **fit_params):
            super(PipelineRFE, self).fit(X, y, **fit_params)
            self.feature_importances_ = self.steps[-1][-1].feature_importances_
            return self
    # Define pipeline for RFE
    steps = [ ("classifier", classifier)]
    pipe = PipelineRFE(steps = steps)

    # Initialize RFE object
    feature_selector = RFE(pipe, n_features_to_select = 10, step = 1, verbose = 1)

    # Fit RFE
    feature_selector.fit(X_train, np.ravel(y_train))

    # Get selected features labels
    feature_names = X_train.columns
    selected_features = feature_names[feature_selector.support_].tolist()
    SelFeatures[i]= selected_features
    print("RFE selected Features", selected_features)

    ###############################################################################
#                  3. Visualizing Selected Features Importance               #
###############################################################################
    # Get selected features data set
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Train classifier
    classifier.fit(X_train, np.ravel(y_train))

    # Get feature importance
    feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])
    feature_importance["Feature Importance"] = classifier.feature_importances_

    # Sort by feature importance
    feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)
    FeatImp.append(feature_importance)
    # Set graph style
    sns.set(font_scale = 1.75)
    sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                   "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                   'ytick.color': '0.4'})

    # Set figure size and create barplot
    f, ax = plt.subplots(figsize=(12, 9))
    sns.barplot(x = "Feature Importance", y = "Feature Label",
                palette = reversed(sns.color_palette('YlOrRd', 15)),  data = feature_importance)

    # Generate a bolded horizontal line at y = 0
    ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

    # Turn frame off
    ax.set_frame_on(False)

    # Tight layout
    plt.tight_layout()

    # Save Figure
    plt.savefig(path+"feature_importance"+str(i)+".png", dpi = 1080)
    

    ###############################################################################
#                       4. Classifier Evaluation                  #
###############################################################################
# Initialize dictionary to store results
    results = {}
    TPredictions = {}
    tPredictions = {}   


    predictions = classifier.predict(X_test)
    auc_te = metrics.roc_auc_score(y_test, predictions)
    
    ## save individual predictions
    data = {'Index': (y_test.index),
            'AET':y_test,
                'Predictions':predictions, 
               "Split":"Test",
               "Iteration":i}
    Ptest = pd.DataFrame(data)
    tPredictions[i] = Ptest

    tr_predictions = classifier.predict(X_train)
    auc_tr = metrics.roc_auc_score(y_train, tr_predictions)

    data = {'Index': (y_train.index),
            'AET':y_train,
                'Predictions':tr_predictions, 
               "Split":"Train",
               "Iteration":i}
    Ptrain = pd.DataFrame(data)
    TPredictions[i] = Ptrain
    # Save results
    result = {"Classifier": classifier,
                  "Best Parameters": best_params,
                  "Training AUC": auc_tr,
                  "Test AUC": auc_te}

    RESULTS[i]=result
    TPREDS[i]=(TPredictions)
    tPREDS[i]=(tPredictions)
    
#################################    
### CV with Selected Features ###
#################################
X = mca_aet[set_features]
y = mca_aet["R2_AET"]

scores_te = list()
scores_tr = list()

feat_imp = list()
ClasRepCV = pd.DataFrame()
feat_imp_RFCV = pd.DataFrame()
PermutResultsRFCV = pd.DataFrame()
tPredictions = {}
TPredictions = {}
i=0

###### Outer Fold Split X_train X_test #######
##################################

out_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

for train_index, test_index in out_split.split(X,y):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    i+=1
    
    model = RandomForestClassifier(bootstrap = True,
                                   oob_score= True,
                                  n_estimators=100,
                                  max_depth=4,
                                  class_weight='balanced',
                                  max_features='sqrt')
    model.fit(X_train,y_train)
    # evaluate model
    yhat = model.predict(X_test)
    tr_predictions = model.predict(X_train)
    
    data = {'Index': (y_test.index),
            'AET':y_test,
                'Predictions':yhat, 
               "Split":"Test",
               "Iteration":i}
    Ptest = pd.DataFrame(data)
    tPredictions[i] = Ptest

    data = {'Index': (y_train.index),
            'AET':y_train,
                'Predictions':tr_predictions, 
               "Split":"Train",
               "Iteration":i}
    Ptrain = pd.DataFrame(data)
    TPredictions[i] = Ptrain
        
    roc_auc = roc_auc_score(y_test, yhat)
    # store score
    scores_te.append(roc_auc)
    
    roc_auc_tr = roc_auc_score(y_train, tr_predictions)
    # store score
    scores_tr.append(roc_auc_tr)
    print('> ', roc_auc)
    cr = (classification_report(y_test, yhat,output_dict=True))
    cr = pd.DataFrame.from_dict(cr).T
    ClasRepCV = pd.concat([ClasRepCV,cr])
    features = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feats = pd.DataFrame(np.flipud([features[z] for z in indices]))
    feats["Importance"] = importances #= (pd.DataFrame(importances))
    feat_imp_RFCV = pd.concat([feat_imp_RFCV,feats.sort_values("Importance", ascending=False)])
    #             Feature Permutation 
    resultp = permutation_importance(model, X_train, y_train, n_repeats=10, n_jobs=-1)
    perm_sorted_idx = resultp.importances_mean.argsort()
    permutations = pd.DataFrame({'Feature':X_train.columns.values[perm_sorted_idx][::-1],'Mean':resultp.importances_mean[perm_sorted_idx][::-1],
                  'std':resultp.importances_std[perm_sorted_idx][::-1]})
    PermutResultsRFCV = pd.concat([PermutResultsRFCV,permutations])

# summarize model performance
mean_s, std_s = mean(scores_te), std(scores_te)
print('Mean: %.3f, Standard Deviation: %.3f' % (mean_s, std_s))
