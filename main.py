from LogReg_utils import *
from sklearn import datasets

# Initialization
print('Cross validation using python libraries')
nFold = 5
nRFE = 10
solver = 'liblinear'
max_iter = 100

# Determine scoring metric
score_metric = ['arccuacy', 'balanced_accuracy', 'average_precision', 'brier_score_loss','f1', 'f1_micro',
                   'f1_macro', 'f1_weighted', 'f1_samples', 'neg_log_loss', 'precision', 'recall', 'jaccard', 'roc_auc']
score_attribute = ['accuracy_score', 'balanced_accuracy_score', 'average_precision_score', 'brier_score_loss',
      'f1_score', 'f1_score', 'f1_score', 'f1_score', 'f1_score', 'log_loss', 'precision_score', 'recall_score',
                     'jaccard_score', 'roc_auc_score']
score_average = ['binary', 'macro', 'micro', 'weighted', 'samples']
idxScore = 10
idxAvg = 1

# Breast cancer prediction
X, y, type = data_prep(datasets.load_breast_cancer())
# X_RFE_train, X_RFE_test, y_RFE_train, y_RFE_test = RFE(X, y, nRFE, solver, max_iter)
cv_results, confusionmatrix, classificationreport = crossvalidate_usinglibrary(X, y, score_metric[idxScore], score_attribute[idxScore],
                                                         score_average[idxAvg], nFold, solver, max_iter)

# iris dataset
X, y, type = data_prep(datasets.load_iris())
cv_results, confusionmatrix, classificationreport = crossvalidate_usinglibrary(X, y, score_metric[idxScore], score_attribute[idxScore],
                                                         score_average[idxAvg], nFold, solver, max_iter)


