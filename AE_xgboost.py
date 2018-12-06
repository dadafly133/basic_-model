#! /usr/bin/python3
import numpy as np
import xgboost as xgb
import scipy.io as sio
from scipy import interp

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, auc, roc_curve
import tensorflow as tf
import time

'''
define function
'''
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)


def apply_fully_connect(x, x_size, fc_size, activition):
	fc_weight = weight_variable([x_size, fc_size])
	fc_bias = bias_variable([fc_size])
	fc = tf.add(tf.matmul(x, fc_weight), fc_bias)
	if activition == "elu":
		return tf.nn.elu(fc)
	elif activition == "relu":
		return tf.nn.relu(fc)
	elif activition == "sigmoid":
		return tf.nn.sigmoid(fc)
	elif activition == "linear":
		return fc
	elif activition == "tanh":
		return tf.nn.tanh(fc)


def multiclass_roc_auc_score(y_true, y_score):
    assert y_true.shape == y_score.shape
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_true.shape[1]
    # compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc


'''
read and process data 
'''
subject_id = 1
data = sio.loadmat("/home/dadafly/program/19AAAI/bci_data/data_folder/cross_sub/cross_subject_data_"+str(subject_id)+".mat")

test_x	= data["test_x"]
train_x	= data["train_x"]

test_y	= data["test_y"].ravel()
train_y = data["train_y"].ravel()

n_channel = test_x.shape[1]
n_length = test_x.shape[2]

test_x = np.reshape(test_x, [-1, n_channel*n_length])
train_x = np.reshape(train_x, [-1, n_channel*n_length])

AE_train = np.vstack([train_x, test_x])
np.random.shuffle(AE_train)

'''
define parameter 
'''
# Training Parameters
learning_rate = 1e-4
num_epoch = 35
batch_size = 30

# Network Parameters
num_input = train_x.shape[1] # input EEG signal shape
num_hidden_1 = 500 # 1st layer num features
num_batch = train_x.shape[0]//batch_size

'''
construct autoencoder
'''
X = tf.placeholder(tf.float32, shape=[None, num_input])

# Construct Encoder
encoder_op = apply_fully_connect(X, num_input, num_hidden_1, "relu")

# Construct Decoder
decoder_op = apply_fully_connect(encoder_op, num_hidden_1, num_input, "linear")

# Prediction
y_pred = decoder_op

# Targets
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

'''
run autoencoder
'''
# run with gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Run the initializer
sess.run(tf.global_variables_initializer())

# Training
for epoch in range(1, num_epoch+1):
	loss_history = np.zeros(shape=[0], dtype=float)
	# training per epoch
	for b in range(num_batch):
		# Prepare data
		offset = (b * batch_size) % (AE_train.shape[0] - batch_size)
		batch_x = AE_train[offset:(offset + batch_size), :]

		# Run optimization op (backprop) and cost op (to get loss value)
		_, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
		loss_history = np.append(loss_history, l)

	# Display logs per epoch 
	print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch, "AE Training loss: ", np.mean(loss_history))


'''
encode data
'''
train_x_en = sess.run(encoder_op, feed_dict={X:train_x})
test_x_en = sess.run(encoder_op, feed_dict={X:test_x})


'''
define xgboost
'''
xg_train = xgb.DMatrix(train_x_en, label=train_y)
xg_test = xgb.DMatrix(test_x_en, label=test_y)
# setup parameters for xgboost
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 4
param['gamma'] = 2
param['subsample'] = 0.5
param['scale_pos_weight'] = 1

num_round = 20

watchlist = [(xg_train, 'train'), (xg_test, 'test')]

'''
train xgboost
'''
bst = xgb.train(param, xg_train, num_round, watchlist)

'''
test and evaluate model
'''
pred_prob = bst.predict(xg_test)
pred_y = np.argmax(pred_prob, axis = -1)

accuracy = np.sum(pred_y==test_y)/test_y.shape[0]

f1=dict()
f1['micro'] = f1_score(y_true=test_y, y_pred=pred_y, average='micro')
f1['macro'] = f1_score(y_true=test_y, y_pred=pred_y, average='macro')

lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)
roc_auc = multiclass_roc_auc_score(y_true = test_y, y_score = pred_prob)

# Printing the results
print("#######################################################################################")
print("subject # ", subject_id)
print("test Classification accuracy:", accuracy)
print("test micro f1:", f1['micro'])
print("test macro f1:", f1['macro'])
print("test micro auc_roc:", roc_auc['micro'])
print("test macro auc_roc:", roc_auc['macro'])
print("#######################################################################################")
