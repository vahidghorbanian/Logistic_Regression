import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
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


def LogReg_BreastCancer(scoring_metric, scoring_attribute, nFold, nRFE, solver, max_iter):
    # Description of dataset
    data = load_breast_cancer()
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
        nonZero = np.count_nonzero(data['target'])
        print('Percentage of nonzero target values:', nonZero*100/len(data['target']))
        print('Percentage of zero target values:', (len(data['target'])-nonZero) * 100 / len(data['target']))
        if (nonZero*100/len(data['target'])) != 50:
            print('The data set is unbalanced!')
    else:
        print('This is a multi-class classification problem!')

    # Data statistics
    print('\n**********************************************************')
    print('Data statistics')
    print(data.describe())

    # Correlation of input features
    X = data.loc[:, data.columns != 'target']
    y = data.loc[:, data.columns == 'target']
    corrMatrix = X.corr(method='pearson')
    # sns.heatmap(corrMatrix.corr(method='pearson'))

    # region Description (Recursive Feature Elimination)
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
    # endregion\

    # region Description (Cross validation)
    print('\n**********************************************************')

    # Using libraries
    print('Cross validation using python libraries')
    print('Number of folds', nFold)
    model = LogisticRegression(max_iter=max_iter, solver=solver)
    y_pred = cross_val_predict(model, X_test, np.ravel(y_test), cv=nFold)
    cv_results = cross_validate(model, X, np.ravel(y), cv=nFold, return_train_score=True, scoring=scoring_metric)
    print('Selected scoring metric: {', scoring_metric, '}')
    print('Fit time:', cv_results['fit_time'])
    print('Score time:', cv_results['score_time'])
    print('Test score:', cv_results['test_score'])
    print('Train score:', cv_results['train_score'])
    print('test score (mean)', cv_results['test_score'].mean())
    print('test score (std)', cv_results['test_score'].std())
    print('confusion matrix:', metrics.confusion_matrix(y_test, y_pred))

    # My own way
    print('\nCross validation my own way')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_kfold, X_test_kfold, y_train_kfold, y_test_kfold = kfold(X, y, nFold) # k-fold split
    test_score = []
    train_score = []
    models_kfold = {}
    models_kfold['test_score'] = []
    models_kfold['train_score'] = []
    models_kfold['fit_time'] = []
    models_kfold['y_pred'] = list()
    for i in np.arange(1, nFold+1, 1):
        model = LogisticRegression(max_iter=max_iter, solver=solver)
        t = time.time()
        models_kfold['f'+str(i)] = model.fit(X_train_kfold['f'+str(i)], np.ravel(y_train_kfold['f'+str(i)]))
        models_kfold['fit_time'] = np.append(models_kfold['fit_time'], time.time() - t)
        models_kfold['y_pred'].append(models_kfold['f' + str(i)].predict(X_test_kfold['f' + str(i)]))
        models_kfold['test_score'] = np.append(models_kfold['test_score'],
                               getattr(metrics, scoring_attribute)(y_test_kfold['f'+str(i)], models_kfold['y_pred'][i-1]))
        models_kfold['train_score'] = np.append(models_kfold['train_score'],
                                getattr(metrics, scoring_attribute)(y_train_kfold['f'+str(i)],
                                                                    models_kfold['f'+str(i)].predict(X_train_kfold['f'+str(i)])))
    print('Fit time:', models_kfold['fit_time'])
    print('Test score:', models_kfold['test_score'])
    print('Train score:', models_kfold['train_score'])
    print('test score (mean)', models_kfold['test_score'].mean())
    print('test score (std)', models_kfold['test_score'].std())
    print('confusion matrix:', metrics.confusion_matrix(y_test, models_kfold['f' + str(np.argmax(models_kfold['test_score']))].predict(X_test)))
    # endregion







