#! /usr/bin/python3

import pandas as pd
import numpy as np

###############################
# read usage and trait data
###############################
big_five_trait_class = pd.read_csv("/home/dadafly/datasets/mobile_phone/big_five_trait_class.csv")
usage_data = pd.read_csv("/home/dadafly/datasets/mobile_phone/usage_com_data.csv")

trait_class_openness 			= big_five_trait_class["openness"]
trait_class_agreeableness 		= big_five_trait_class["agreeableness"]
trait_class_conscientiousness 	= big_five_trait_class["conscientiousness"]
trait_class_extraversion 		= big_five_trait_class["extraversion"]
trait_class_neuroricism 		= big_five_trait_class["neuroricism"]

#print(usage_data.shape)
#print(usage_data.head())
#print(np.max(trait_openness), np.min(trait_openness), np.mean(trait_openness))

###############################
# train test split 
###############################
from sklearn.cross_validation import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(usage_data, trait_class_openness,			test_size=0.25, random_state=33)
# X_train, X_test, y_train, y_test = train_test_split(usage_data, trait_class_agreeableness, 	 	test_size=0.25, random_state=33)
# X_train, X_test, y_train, y_test = train_test_split(usage_data, trait_class_conscientiousness,	test_size=0.25, random_state=33)
# X_train, X_test, y_train, y_test = train_test_split(usage_data, trait_class_extraversion, 	 	test_size=0.25, random_state=33)
X_train, X_test, y_train, y_test = train_test_split(usage_data, trait_class_neuroricism,		test_size=0.25, random_state=33)

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
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=80, criterion="gini", max_depth=40, random_state=33)

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

evaluate_cross_validation(clf_rf, X_train, y_train, 5)

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

train_and_evalutate(clf_rf, X_train, X_test, y_train, y_test)
