'''
implementation of basic ML models:
decision tree
random forest
svm
knn
naive bayes
'''

import pandas as pd
import numpy as np

###############################
# read data
###############################
features = pd.read_csv("~/features.csv")
labels = pd.read_csv("~/labels.csv")

###############################
# train test split 
###############################
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=33)

###############################
# normalization  
###############################
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(X_train)

X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

###############################
# define model  
###############################
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

def select_model(model_name):
	if model == "dt":
		model = tree.DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=5)
	elif model == "rf":
		model = RandomForestClassifier(n_estimators=80, criterion="gini", max_depth=40, random_state=33)
	elif model == "knn":
		model = KNeighborsClassifier(n_neighbors=30)
	elif model == "svm":
		model = SVC(kernel='rbf', class_weight={1:0.3}, C=500)#, gamma=0.001)# conscientiousneww{1:0.3} agreeableness{1:0.4}
	elif model == "nb":
		# model = MultinomialNB(alpha=0.0)
		# model = GaussianNB()
		model = BernoulliNB()
	else:
		print("Ivalid model name!")
		exit()
	return model

clf = select_model(model_name='dt')

###############################
# cross validation evaluate
###############################
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X_train, y_train, K):
	# create a k-fold cross validation iterator of k=5 folds
	cv = KFold(X_train.shape[0], K, shuffle=True, random_state=33)
	scores = cross_val_score(clf, X_train, y_train, cv=cv)
	print(scores)
	print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

evaluate_cross_validation(clf, X_train, y_train, 5)

###############################
# train and evaluate
###############################
from sklearn import metrics

def train_and_evalutate(clf, X_train, X_test, y_train, y_test):
	clf.fit(X_train, y_train)
	
	print("Accuracy on training set:")
	print(clf.score(X_train, y_train))
	print("Accuracy on testing set:")
	print(clf.score(X_test, y_test))

	y_pred = clf.predict(X_test)

	print("Classification Report:")
	print(metrics.classification_report(y_test, y_pred))
	print("Confusion Matrix:")
	print(metrics.confusion_matrix(y_test, y_pred))

train_and_evalutate(clf, X_train, X_test, y_train, y_test)
