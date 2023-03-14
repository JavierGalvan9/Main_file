import argparse
import csv
import glob
import os
import random
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from callbacks import MyCallback
from data_preprocessing import data_preprocessing
from models import L2_regularized_model, NN_model

parentDir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management

seed(0)

print(tf.config.list_physical_devices('GPU'))
# Use the first available GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Only use the memory needed
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# Load the training and test sets
train_test_path = os.path.join('Classification datasets', 'Train and test sets')
X_train, X_test, y_train, y_test, std_scale, cluster_id_train, cluster_id_test = data_preprocessing(train_test_path, use_spectral_bands=True, use_indices=True)
# Number of features
print('X_train shape:', X_train.shape)
n_features = X_train.shape[1]
print('Number of features:', n_features)

# Load the best classifier and the validation set of the best fold 
best_model = load_model('ann_best_classifier.h5')

# =============================================================================
# EVALUATION OF THE BEST CLASSIFIER OVER TEST SET
# =============================================================================
# We evaluate the best classifier on the test set
y_test_pred = best_model.predict(X_test)
Y_test_pred = np.round(y_test_pred)
# Compute the confusion matrix
cm=confusion_matrix(y_test,Y_test_pred)
# Calculate the accuracy, recall, precision, f1-score and AUC
accuracy = accuracy_score(y_test, Y_test_pred)
recall = recall_score(y_test, Y_test_pred)
precision = precision_score(y_test, Y_test_pred)
f1score = f1_score(y_test, Y_test_pred)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_test_pred)
auc_keras = auc(fpr_keras, tpr_keras)
# Change writing in the file by printing the results
print('------------------------------------------------------------------------')
print('Confussion matrix over test set')
print('------------------------------------------------------------------------')
print(cm)
print('------------------------------------------------------------------------')
print("Accuracy: %.2f%%" % (accuracy*100))
print("Recall: %.2f%%" % (recall*100))
print("Precision: %.2f%%" % (precision*100))
print("F1-score: %.2f%%" % (f1score*100))
print("AUC: %.2f" % (auc_keras))
print("Proportion of positives: %.2f%%" % (100*sum(y_test)/len(y_test)))
print('------------------------------------------------------------------------')

# =============================================================================
# ROC and AUC
# =============================================================================
fig = plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='ANN 1-1 (AUC = {:.2f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate (specificity)', fontsize=16)
plt.ylabel('True positive rate (sensitivity)', fontsize=16)
plt.legend(loc='best', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.close()

# Repeat the previous analysis over the training set
y_train_pred = best_model.predict(X_train)
Y_train_pred = np.round(y_train_pred)
# Compute the confusion matrix
cm=confusion_matrix(y_train,Y_train_pred)
# Calculate the accuracy, recall, precision, f1-score and AUC
accuracy = accuracy_score(y_train, Y_train_pred)
recall = recall_score(y_train, Y_train_pred)
precision = precision_score(y_train, Y_train_pred)
f1score = f1_score(y_train, Y_train_pred)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_train, y_train_pred)
auc_keras = auc(fpr_keras, tpr_keras)
# Change writing in the file by printing the results
print('------------------------------------------------------------------------')
print('Confussion matrix over training set')
print('------------------------------------------------------------------------')
print(cm)
print('------------------------------------------------------------------------')
print("Accuracy: %.2f%%" % (accuracy*100))
print("Recall: %.2f%%" % (recall*100))
print("Precision: %.2f%%" % (precision*100))
print("F1-score: %.2f%%" % (f1score*100))
print("AUC: %.2f" % (auc_keras))
print("Proportion of positives: %.2f%%" % (100*sum(y_train)/len(y_train)))
print('------------------------------------------------------------------------')