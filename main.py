from LogReg_utils import *

print('\n***Classification problem: Breast classification***')
print('\nScoring metric used for classification can be one of the following:\n'
      '{arccuacy,', 'balanced_accuracy,', 'average_precision,', 'brier_score_loss,\n',
      'f1,', 'f1_micro,', 'f1_macro,', 'f1_weighted,', 'f1_samples,', 'neg_log_loss,\n',
      'precision,', 'recall,', 'jaccard,', 'roc_auc}\n')
print('The associated attribute names in python under "metrics" are:\n'
      '{accuracy_score,', 'balanced_accuracy_score,', 'average_precision_score,', 'brier_score_loss,\n',
      'f1_score,', 'f1_score,', 'f1_score,', 'f1_score,', 'f1_score,', 'log_loss,\n',
      'precision_score,', 'recall_score,', 'jaccard_score,', 'roc_auc_score}')

scoring_metric = 'f1'
scoring_attribute = 'f1_score'
nFold = 5
nRFE = 10
solver = 'liblinear'
max_iter = 100
LogReg_BreastCancer(scoring_metric, scoring_attribute, nFold, nRFE, solver, max_iter)
