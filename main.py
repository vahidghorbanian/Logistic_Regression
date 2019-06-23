from LogReg_utils import *
from sklearn import datasets

# Initialization
print('Cross validation using python libraries')
nRFE = 10
solver = 'liblinear'
max_iter = 10000
penalty = 'l2'
cv = True # cross validation
nFold = 5

# Determine scoring metric
scoring_metric = None

# Breast cancer prediction (binary classification)
X, y, type = data_prep(datasets.load_breast_cancer())
# X_RFE_train, X_RFE_test, y_RFE_train, y_RFE_test = RFE(X, y, nRFE, solver, max_iter)
confusionmatrix, classificationreport, best_model = crossvalidate_usinglibrary(X, y, nFold, solver, max_iter,
                                                                               scoring_metric, penalty, cv)

# iris dataset (Multiclass classification)
X, y, type = data_prep(datasets.load_iris())
confusionmatrix, classificationreport, best_model = crossvalidate_usinglibrary(X, y, nFold, solver, max_iter,
                                                                               scoring_metric, penalty, cv)





