# Logistic_Regression

In the context of machine learning, logistic regression is a classification tool, not a regression used for contineous data sets.
Basically, the linear regression's hypothesis (y = b + a1x1 + a2x2 + ...) is converted to a nonlinear hypothesis as follows:

h(X) = 1 / (1+exp(-y)),

where h(X) is a sigmoid function varying from 0 to one in a sinusoidal shape. 

* Logistic regression can work with both continuous and discrete data.
* Logisitic regression does not have the same concept of a "residual", so it can't use least square and it can't calculate R^2
