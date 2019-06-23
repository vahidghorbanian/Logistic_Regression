import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import metrics
import time
import sys


def kfold(X, y, nFold):
    print('K-fold split')
    print('Number of folds:', nFold)
    kf = KFold(n_splits=nFold, shuffle=True, random_state=None)
    X_train = {}
    X_test  = {}
    y_train = {}
    y_test  = {}
    print('The train and test kfold sets are stored in dictionaries.\n'
          'For example, X_train{f1:}, X_train{f2:} and so on.')
    k = 1
    for train_index, test_index in kf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train['f' + str(k)], X_test['f' + str(k)] = X.loc[train_index, :], X.loc[test_index, :]
        y_train['f' + str(k)], y_test['f' + str(k)] = y.loc[train_index, :], y.loc[test_index, :]
        print('Size of fold ', k, ': X_train =', X_train['f' + str(k)].shape, ', y_train =',
              y_train['f' + str(k)].shape,
              ', X_test =', X_test['f' + str(k)].shape, ', y_test =', y_test['f' + str(k)].shape)
        k = k + 1
    return X_train, X_test, y_train, y_test


def RFE(X, y, nRFE, solver, max_iter):
    print('\n**********************************************************')
    print('Recursive Feature Elimination')
    nFeature = nRFE
    if nFeature > len(X.columns):
        sys.exit('\n\nError: Number of RFE features must be equal or less than the data features!')
    elif nFeature < 1:
        sys.exit('\n\nError: Number of RFE features must be larger than 0!')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    score = []
    idxMin = []
    elimFeatureName = []
    print('Total number of features: ', len(X.columns), '\n')
    for count, i in enumerate(np.arange(len(X.columns), nFeature, -1)):
        model = LogisticRegression(max_iter=max_iter, solver=solver)
        model.fit(X_train, np.ravel(y_train))
        score = np.append(score, model.score(X_test, np.ravel(y_test)))
        idxMin = np.append(idxMin, np.argmin(np.abs(model.coef_)))
        elimFeatureName = np.append(elimFeatureName, X_train.columns[idxMin[count]])
        X_train = X_train.drop(X_train.columns[idxMin[count]], axis=1)
        X_test = X_test.drop(X_test.columns[idxMin[count]], axis=1)
        print('Eliminated feature: ', elimFeatureName[count],
              '(Coefficient = ', round(model.coef_[0, int(idxMin[count])], 4), ')')
    print('\nNumber of remaining features:', nFeature)
    model = LogisticRegression(max_iter=max_iter, solver=solver)
    model.fit(X_train, np.ravel(y_train))
    print('Model score: ', model.score(X_test, y_test), '\n')
    coef = model.coef_
    feature = list(X_train.columns)
    print('Remaining features:\n')
    for count, i in enumerate(np.arange(0, len(feature), 1)):
        idxMax = np.argmax(np.abs(coef))
        print(feature[idxMax],'(Coefficient = ', round(float(np.transpose(coef)[idxMax]), 4),')')
        coef = np.delete(coef, idxMax)
        feature = np.delete(feature, idxMax)
    X_RFE_train = X_train
    X_RFE_test  = X_test
    y_RFE_train = y_train
    y_RFE_test  = y_test
    return X_RFE_train, X_RFE_test, y_RFE_train, y_RFE_test


def crossvalidate_usinglibrary(X, y, nFold, solver, max_iter, scoring_metric, penalty, cv):
    print('\n**********************************************************')
    print('Number of folds', nFold)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    C = [0.1, 1, 5, 10, 50, 100]
    tuned_parameters = {'C': C}
    if cv == True:
        model = LogisticRegression(max_iter=max_iter, solver=solver, multi_class='auto', random_state=0,
                                   penalty=penalty)
        clf = GridSearchCV(model, tuned_parameters, cv=nFold, refit=True, return_train_score=False, scoring=scoring_metric)
        models = clf.fit(X_train, np.ravel(y_train))
        best_model = models.best_estimator_
    else:
        score = 0
        for i in np.arange(0, len(C), 1):
            model = LogisticRegression(max_iter=max_iter, solver=solver, multi_class='auto', random_state=0,
                                       penalty=penalty, C=C[i])
            model.fit(X_train, np.ravel(y_train))
            if model.score(X_train, np.ravel(y_train)) > score:
                score = model.score(X_train, np.ravel(y_train))
                best_model = model
    y_pred = best_model.predict(X_test)
    confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
    classificationreport = metrics.classification_report(y_test, y_pred)
    # print('\n', models.cv_results_, '\n')
    print('confusion matrix:', confusionmatrix)
    print('classification report:', classificationreport)
    return confusionmatrix, classificationreport, best_model


def crossvalidation_myway(X, y, scoring_attribute, nFold, solver, max_iter):
    print('\n**********************************************************')
    print('Cross validation my own way')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_kfold, X_test_kfold, y_train_kfold, y_test_kfold = kfold(X, y, nFold) # k-fold split
    cv_results = {}
    cv_results['test_score'] = []
    cv_results['train_score'] = []
    cv_results['fit_time'] = []
    cv_results['y_pred'] = list()
    for i in np.arange(1, nFold+1, 1):
        model = LogisticRegression(max_iter=max_iter, solver=solver)
        t = time.time()
        cv_results['f'+str(i)] = model.fit(X_train_kfold['f'+str(i)], np.ravel(y_train_kfold['f'+str(i)]))
        cv_results['fit_time'] = np.append(cv_results['fit_time'], time.time() - t)
        cv_results['y_pred'].append(cv_results['f' + str(i)].predict(X_test_kfold['f' + str(i)]))
        cv_results['test_score'] = np.append(cv_results['test_score'],
                               getattr(metrics, scoring_attribute)(y_test_kfold['f'+str(i)], cv_results['y_pred'][i-1]))
        cv_results['train_score'] = np.append(cv_results['train_score'],
                                getattr(metrics, scoring_attribute)(y_train_kfold['f'+str(i)],
                                                                    cv_results['f'+str(i)].predict(X_train_kfold['f'+str(i)])))
    print('Fit time:', cv_results['fit_time'])
    print('Test score:', cv_results['test_score'])
    print('Train score:', cv_results['train_score'])
    print('test score (mean)', cv_results['test_score'].mean())
    print('test score (std)', cv_results['test_score'].std())
    confusionmatrix = metrics.confusion_matrix(y_test, cv_results['f' + str(np.argmax(cv_results['test_score'])+1)].predict(X_test))
    print('confusion matrix:', confusionmatrix)
    return cv_results, confusionmatrix


def data_prep(data):
    # data = datasets.load_breast_cancer()
    data = pd.DataFrame(data=np.append(data['data'], np.transpose([data['target']]), axis=1),
                        columns=np.append(data['feature_names'], ['target']))
    print('\n**********************************************************')
    print('Feature names:\n', list(data.columns[0:len(data.columns)-1]))
    print('\nTarget name:\n', data.columns[len(data.columns)-1])
    print('\n**********************************************************')
    print('Data type:\n', data.dtypes)
    print('\n**********************************************************')
    print('Unique values of output:\n', np.unique(data['target']))
    if len(np.unique(data['target'])) == 2:
        print('This is a binary classification problem!')
        type = 'binary'
        nonZero = np.count_nonzero(data['target'])
        print('Percentage of nonzero target values:', nonZero*100/len(data['target']))
        print('Percentage of zero target values:', (len(data['target'])-nonZero) * 100 / len(data['target']))
        if (nonZero*100/len(data['target'])) != 50:
            print('The data set is unbalanced!')
    else:
        print('This is a multi-class classification problem!')
        type = 'multiclass'
    # Data statistics
    print('\n**********************************************************')
    print('Data statistics')
    print(data.describe())
    # Correlation of input features
    X = data.loc[:, data.columns != 'target']
    y = data.loc[:, data.columns == 'target']
    corrMatrix = X.corr(method='pearson')
    mask = np.zeros_like(corrMatrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corrMatrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.show()
    return X, y, type



