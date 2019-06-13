import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Main functions
data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
print(data.head())
print(data['education'].unique())

# group “basic.4y”, “basic.9y” and “basic.6y” together and call them “basic”
data['education'] = np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education'] =='basic.4y', 'Basic', data['education'])
print(data['education'].unique())

# Data exploration
print(data['y'].value_counts())
# sns.countplot(x='y', data=data, palette='hls')
# plt.show()
count_sub = np.count_nonzero(data['y'])
count_no_sub = len(data['y']) - count_sub
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("\npercentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)
a = data.groupby('marital').mean()
print('\n', a)

# matplotlib inline
# table = pd.crosstab(data.job, data.education)
# print('\n', table)
# table.plot(kind='bar')
# plt.title('Purchase Frequency for Job Title')
# plt.xlabel('Job')
# plt.ylabel('Frequency of Purchase')

# table = pd.crosstab(data.education,data.y)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# plt.title('Stacked Bar Chart of Education vs Purchase')
# plt.xlabel('Education')
# plt.ylabel('Proportion of Customers')
# plt.savefig('edu_vs_pur_stack')

# Create dummy variables
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(cat_list)
    data = data1
cat_vars = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars = data.columns.values.tolist()
to_keep = [i for i in data_vars if i not in cat_vars]
data_final = data[to_keep]

# Over-sampling using SMOTE
''' This part of the function over-samples the minority member of data_final['y']. 
Works by creating synthetic samples from the minor class (no-subscription) instead of creating copies.
Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observations.
'''
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
os = SMOTE(random_state=0, sampling_strategy = 'auto', ratio=None)
print('\nSMOTE settings are as follows:\n', os)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
# we can Check the numbers of our data
print("\nlength of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

# Recursive Feature Elimination
" Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and" \
" choose either the best or worst performing feature, setting the feature aside and then repeating" \
" the process with the rest of the features. This process is applied until all features in the dataset" \
" are exhausted. The goal of RFE is to select features by recursively considering smaller and smaller sets of features."
data_final_vars = data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
logreg = LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=1.0, fit_intercept=True, intercept_scaling=1,
                            class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='warn',
                            verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
numFeature = 20
X_test = pd.DataFrame(data=X_test, columns=columns)
y_test = pd.DataFrame(data=y_test, columns=['y'])
score = []
k = []
# My own way
print('\n********Recursive Feature Elimination********')
for count, var in enumerate(X.columns):
    model = logreg.fit(X, np.ravel(y))
    y_predicted = model.predict(X_test)
    score.append(logreg.score(X_test, y_test))
    idxMin = np.argmin(np.abs(model.coef_))
    X = X.drop(X.columns[idxMin], axis=1)
    X_test = X_test.drop(X_test.columns[idxMin], axis=1)
    print('Number of remaining features: ', X.shape[1])
    k.append(X.shape[1])
    if X.shape[1] <= numFeature:
        break
# Plot results of Recursive feature elimination
f = plt.figure()
plt.plot(k, score)
plt.xlabel('Number of features')
plt.ylabel('Logistic Regression model accuracy')
# Built-in function
# rfe = RFE(logreg, n_features_to_select=20, step=1, verbose=0)
# rfe = rfe.fit(os_data_X, os_data_y)
# os_data_y_predict = rfe.predict(os_data_X)
# RMSE = mean_squared_error(os_data_y_predict,os_data_y)
# print(rfe.support_)
# print(rfe.ranking_)
# Features with highest rank
# cols = ['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown',
#       'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
#       'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"]
# X = os_data_X[cols]
# y = os_data_y['y']


plt.show()