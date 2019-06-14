import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFE
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import sys
from imblearn.over_sampling import SMOTE

def kfold(X, y, nFold):
    print('K-fold split')
    print('number of folds:', nFold)
    kf = KFold(n_splits=nFold, shuffle=False, random_state=0)
    X_train = {}
    X_test  = {}
    y_train = {}
    y_test  = {}
    print('The train and test kfold sets are stored in dictionaries.\n'
          'For example, X_train{F1:}, X_train{F2:} and so on.')
    k = 1
    for train_index, test_index in kf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train['F' + str(k)], X_test['F' + str(k)] = X.loc[train_index, :], X.loc[train_index, :]
        y_train['F' + str(k)], y_test['F' + str(k)] = y.loc[test_index, :], y.loc[test_index, :]
        print('Size of fold ', k, ': X_train =', X_train['F' + str(k)].shape, ', y_train =',
              y_train['F' + str(k)].shape,
              ', X_test =', X_test['F' + str(k)].shape, ', y_test =', y_test['F' + str(k)].shape)
        k = k + 1
    return X_train, X_test, y_train, y_test


def LogReg_BreastCancer():
    # Description of dataset
    print('\n***Classiifcation problem: Breast classification.***')
    data = load_breast_cancer()
    data = pd.DataFrame(data=np.append(data['data'], np.transpose([data['target']]), axis=1),
                        columns=np.append(data['feature_names'], ['target']))
    print('\n**********************************************************')
    print('Feature names:\n', data.columns[0:len(data.columns)-1])
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
            print('The data set is imbalanced!')
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
    sns.heatmap(corrMatrix.corr(method='pearson'))

    # region Description (Recursive Feature Elimination)
    print('\n**********************************************************')
    print('Recursive Feature Elimination')
    nFeature = 10
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
        model = LogisticRegression(max_iter=100, solver='liblinear')
        model.fit(X_train, np.ravel(y_train))
        score = np.append(score, model.score(X_test, np.ravel(y_test)))
        idxMin = np.append(idxMin, np.argmin(np.abs(model.coef_)))
        elimFeatureName = np.append(elimFeatureName, X_train.columns[idxMin[count]])
        X_train = X_train.drop(X_train.columns[idxMin[count]], axis=1)
        X_test = X_test.drop(X_test.columns[idxMin[count]], axis=1)
        print('Eliminated feature: ', elimFeatureName[count],
              '\nFeatures coefficient:', round(model.coef_[0, int(idxMin[count])], 4))
    print('\nNumber of remaining features:', nFeature)

    model = LogisticRegression(max_iter=100, solver='liblinear')
    model.fit(X_train, np.ravel(y_train))
    print('Model score: ', model.score(X_test, y_test))
    coef = model.coef_
    feature = list(X_train.columns)
    for count, i in enumerate(np.arange(0, len(feature), 1)):
        idxMax = np.argmax(np.abs(coef))
        print('Feature: ', feature[idxMax],
              '\nFeature coefficient:', round(float(np.transpose(coef)[idxMax]), 4))
        coef = np.delete(coef, idxMax)
        feature = np.delete(feature, idxMax)
    X_RFE_train = X_train
    X_RFE_test  = X_test
    y_RFE_train = y_train
    y_RFE_test  = y_test
    # endregion\

    # region Description (K-fold split)
    print('\n**********************************************************')
    nFold = 5
    kfold(X, y, nFold)
    # plt.show()
    # endregion






