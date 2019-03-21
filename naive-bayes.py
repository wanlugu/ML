############# tranining, testing split
sklearn.model_selection.train_test_split(*arrays, **options)
################### Genralized linear model #########
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0,0,0],[1,1,1],[2,2,2]],[0,1,2])
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
reg.coef_

# ridge regression
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0,0,0],[1,1,1],[2,2,2]],[0,1,2])
Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
# ridge CV
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
reg.fit([[0,0,0],[1,1,1],[2,2,2]],[0,1,2])
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3, fit_intercept=True, scoring=None,
    normalize=False)






################### Nearest Neighbors ###############
################### Naive Bayes ##################
import sklearn
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d point: %d"%(iris.data.shape[0],(iris.target != y_pred).sum()))


