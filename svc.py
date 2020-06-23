#! /usr/bin/python3

import pandas as pd
import numpy as np

###############################
# read usage and trait data
###############################
big_five_trait_class = pd.read_csv("/home/dadafly/datasets/mobile_phone/big_five_trait_class.csv")
usage_data = pd.read_csv("/home/dadafly/datasets/mobile_phone/usage_com_data.csv")

trait_class_openness = big_five_trait_class["openness"]

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
usage_data_reduced = svd.fit_transform(usage_data)
###############################
# train test split 
###############################
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(usage_data_reduced, trait_class_openness, test_size=0.25, random_state=33)

###############################
# normalization  
###############################
from sklearn.preprocessing import StandardScaler
#scalerX = StandardScaler().fit(X_train)
#
#X_train = scalerX.transform(X_train)
#X_test = scalerX.transform(X_test)

###############################
# define model  
###############################
from sklearn.svm import SVC
clf_svc = SVC(kernel='rbf', class_weight={1:0.3}, C=500)#, gamma=0.001)# conscientiousneww{1:0.3} agreeableness{1:0.4}
# clf_svc = SVC(kernel='linear',	class_weight={1:0.3}, C=5, gamma=0.001)
# clf_svc = SVC(kernel='poly',	class_weight={1:0.3}, C=5, gamma=0.001, degree=4)
# clf_svc = SVC(kernel='sigmoid',	class_weight={1:0.3}, C=5, gamma=0.001)

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

evaluate_cross_validation(clf_svc, X_train, y_train, 5)

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

train_and_evalutate(clf_svc, X_train, X_test, y_train, y_test)
