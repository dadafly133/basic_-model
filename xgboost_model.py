#! /usr/bin/python3

import pandas as pd
import numpy as np
import xgboost as xgb


###############################
# read usage and trait data
###############################
big_five_trait_class = pd.read_csv("/home/dadafly/datasets/mobile_phone/big_five_trait_class.csv")
usage_data = pd.read_csv("/home/dadafly/datasets/mobile_phone/usage_com_data.csv")

big_five_trait_class = big_five_trait_class.replace("low", 0)
# big_five_trait_class = big_five_trait_class.replace("moderate", 1)
big_five_trait_class = big_five_trait_class.replace("high", 1)

trait_class_openness 			= np.asarray(big_five_trait_class["openness"])
trait_class_agreeableness 		= np.asarray(big_five_trait_class["agreeableness"])
trait_class_conscientiousness 	= np.asarray(big_five_trait_class["conscientiousness"])
trait_class_extraversion 		= np.asarray(big_five_trait_class["extraversion"])
trait_class_neuroricism 		= np.asarray(big_five_trait_class["neuroricism"])

#print(usage_data.shape)
#print(usage_data.head())
#print(np.max(trait_openness), np.min(trait_openness), np.mean(trait_openness))
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=8, n_iter=5, random_state=42)
usage_data_reduced = svd.fit_transform(usage_data)
###############################
# train test split 
###############################
from sklearn.cross_validation import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(usage_data_reduced, trait_class_openness,			test_size=0.25, random_state=33)
# train_X, test_X, train_Y, test_Y = train_test_split(usage_data, trait_class_agreeableness, 	 	test_size=0.25, random_state=33)
# train_X, test_X, train_Y, test_Y = train_test_split(usage_data, trait_class_conscientiousness,	test_size=0.25, random_state=33)
# train_X, test_X, train_Y, test_Y = train_test_split(usage_data, trait_class_extraversion, 	 	test_size=0.25, random_state=33)
# train_X, test_X, train_Y, test_Y = train_test_split(usage_data, trait_class_neuroricism,		test_size=0.25, random_state=33)

###############################
# normalization  
###############################
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(train_X)

train_X = scalerX.transform(train_X)
test_X = scalerX.transform(test_X)

###############################
# define model  
###############################
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 2
param['gamma'] = 2
param['subsample'] = 0.5
param['scale_pos_weight'] = 1

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 100
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_Y, pred))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
#pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 2)
#pred_label = np.argmax(pred_prob, axis=1)
#error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
#print('Test error using softprob = {}'.format(error_rate))

