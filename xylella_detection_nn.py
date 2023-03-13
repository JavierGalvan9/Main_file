# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:59:22 2022

@author: UX325
"""

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

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", 
                    type=int,
                    default=200,
                    help="especify directory to save data")

parser.add_argument("--oversampling", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-oversampling", dest="oversampling", action="store_false")

parser.add_argument("--undersampling", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-undersampling", dest="undersampling", action="store_false")

parser.add_argument("--cost_sensitive", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-cost_sensitive", dest="cost_sensitive", action="store_false")

parser.add_argument("--dropout", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-dropout", dest="dropout", action="store_false")

parser.add_argument("--use_spectral_bands", 
                    action="store_true",
                    default=True,
                    help="Choose whether to use only spectral bands or not")
parser.add_argument("--no-use_spectral_bands", dest="use_spectral_bands", action="store_false")

parser.add_argument("--use_indices", 
                    action="store_true",
                    default=True,
                    help="Choose whether to use only indices or not")
parser.add_argument("--no-use_indices", dest="use_indices", action="store_false")

parser.add_argument("--L2_regularizer", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-L2_regularizer", dest="L2_regularizer", action="store_false")

args = parser.parse_args()

## Create dir to save classification results and filename based on arguments
classification_main_path = 'Classification results'
args_path = ''
for arg, value in vars(args).items():
    if value:
        args_path += f'{arg}_{value} '

# If there are no arguments, use the default name
if len(args_path) > 0:
    classification_path = os.path.join(classification_main_path, args_path)
else:
    classification_path = os.path.join(classification_main_path, 'Baseline')

# If the directory already exists, add a number to the end of the name
fileCounter = len(glob.glob1(classification_main_path,args_path+'*'))
if fileCounter > 0:
    classification_path += str(fileCounter)
os.makedirs(classification_path, exist_ok=True)

# train_test_path = os.path.join('Classification datasets', 'Train and test sets')
# X_train, X_test, y_train, y_test, std_scale = data_preprocessing(train_test_path, use_spectral_bands=True, use_indices=True)

# Load the training and test sets
train_test_path = os.path.join('Classification datasets', 'Train and test sets')
X_train, X_test, y_train, y_test, std_scale, cluster_id_train, cluster_id_test = data_preprocessing(train_test_path, use_spectral_bands=args.use_spectral_bands, use_indices=args.use_indices)
# Number of features
print('X_train shape:', X_train.shape)
n_features = X_train.shape[1]
print('Number of features:', n_features)

## Train simple NN with 7-fold cross-validation
# Choose the model architecture based on the arguments passed to the script 
if args.L2_regularizer:
    classifier = L2_regularized_model(n_features)
else:
    classifier = NN_model(n_features)
        
# Define the callbacks to use during the training         
callbacks = [
    MyCallback(monitor='val_auc', value=0.95, verbose=1), 
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000) # Stop training when a monitored quantity has stopped improving.
]

# Define the sampling method to balance the training set  
sampling = None
if args.oversampling and not args.undersampling:
    # Oversample the training set using ADASYN 
    from imblearn.over_sampling import ADASYN
    sampling = ADASYN(random_state=42)
if args.undersampling and not args.oversampling:
    # Undersample the training set using TomekLinks
    from imblearn.under_sampling import TomekLinks
    sampling = TomekLinks(sampling_strategy='majority')
if args.oversampling and args.undersampling:
    # Combine over- and undersampling using SMOTE and Tomek links
    from imblearn.combine import SMOTETomek 
    sampling = SMOTETomek(random_state=42)

#Set the training parameters 
num_epochs = args.num_epochs
verbosity = 0 #0:silent, 1:to show a progress bar during the training, 2:show results after each epoch
batch_size = 32
K = 7 # Number of folds for cross-validation
# Define per-fold score containers
acc_per_fold = []
auc_per_fold = []
loss_per_fold = []
ground_auc = 0
fold_no = 1
best_model = None
# K-fold Cross Validation model evaluation 
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=0) #fix the divisions of the data
for train, validate in kfold.split(X_train, y_train):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # Divide the training set into training and validation sets for this fold
    X_train_fold, y_train_fold = X_train[train], y_train[train]
    X_val, y_val = X_train[validate], y_train[validate]
    # Define the number of training examples
    n_training_examples = X_train_fold.shape[0]
    # If sampling is not None, apply it to the training set
    if sampling is not None:
        X_train_fold, y_train_fold = sampling.fit_resample(X_train_fold, y_train_fold)
    ## Fit data to model   
    # If cost-sensitive learning is enabled, compute the class weights and use them to train the model 
    if args.cost_sensitive:        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(class_weight = "balanced",
                                            classes = np.unique(y_train_fold),
                                            y = y_train_fold)
        class_weights = dict(zip(np.unique(y_train), class_weights))
        # Train the model with the class weights
        history = classifier.fit(X_train_fold, y_train_fold, 
                                  validation_data = (X_val, y_val), 
                                  class_weight=class_weights,
                                  batch_size = batch_size, 
                                  epochs = num_epochs, 
                                  verbose=verbosity,
                                  callbacks=callbacks)
    else:
        # Train the model
        history = classifier.fit(X_train_fold, y_train_fold, 
                                validation_data = (X_val, y_val), 
                                batch_size = batch_size, #n_training_examples, 
                                epochs = num_epochs, 
                                verbose=verbosity,
                                callbacks=callbacks)
                                # use_multiprocessing=True)
    # Save the results of this fold into lists for later use
    scores = classifier.evaluate(X_val, y_val, verbose=0)
    loss_per_fold.append(scores[0])
    acc_per_fold.append(scores[1] * 100)
    auc_per_fold.append(scores[2])
    # If the AUC of this fold is the best so far, save the model and the validation set
    if scores[2]>ground_auc:
        ground_auc = scores[2]
        validate_set = validate
        best_model = classifier
        # Save the training history
        train_acc2 = history.history['accuracy']
        val_acc2 = history.history['val_accuracy']
        train_auc2 = history.history['auc']
        val_auc2 = history.history['val_auc']
        train_loss2 = history.history['loss']
        val_loss2 = history.history['val_loss']
    # Increase fold number
    fold_no = fold_no + 1

# Save the best model and the validation set
best_model.save(os.path.join(classification_path, 'ann_best_classifier.h5'))
file_management.save_lzma(validate_set, 'ann_best_fold_validation_set.lzma', classification_path)
# Save the results of the cross-validation 
with open(os.path.join(classification_path, 'training_info.txt'), 'w') as f:
    f.write('------------------------------------------------------------------------ \n')
    f.write('Score per fold \n')
    for i in range(0, len(acc_per_fold)):
        f.write('------------------------------------------------------------------------ \n')
        f.write(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}% - AUC: {auc_per_fold[i]}\n')
    f.write('------------------------------------------------------------------------ \n')
    f.write('Average scores for all folds: \n')
    f.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)}) \n')
    f.write(f'> AUC: {np.mean(auc_per_fold)} (+- {np.std(auc_per_fold)}) \n')
    f.write(f'> Loss: {np.mean(loss_per_fold)} (+- {np.std(loss_per_fold)}) \n')
    f.write('------------------------------------------------------------------------ \n')
    
# =============================================================================
# DRAW THE LEARNING CURVES FOR THE BEST K-FOLD
# =============================================================================
epochs = np.arange(1, len(train_acc2)+1)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(9, 5), sharex=True)
# Draw accuracy curve 
ax1.plot(epochs, train_acc2, 'r-', label='Training set')
ax1.plot(epochs, val_acc2, 'b-', label='Cross-validation set')
ax1.axhline(1- sum(y_train[validate_set])/len(y_train[validate_set]), label='Negatives ratio', color='k')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.legend(loc='best', fontsize=12)
#Draw auc curve
ax2.plot(epochs, train_auc2, 'r-', label='Training set')
ax2.plot(epochs, val_auc2, 'b-', label='Cross-validation set')
ax2.set_ylabel('AUC', fontsize=12)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.legend(loc='best', fontsize=12)
#Draw loss curve 
ax3.plot(epochs, train_loss2, 'r-', label='Training set')
ax3.plot(epochs, val_loss2, 'b-', label='Cross-validation set')
ax3.set_xlabel('Epochs', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.legend(loc='best', fontsize=12)
# Save the figure 
plt.tight_layout()
plt.savefig(os.path.join(classification_path, 'ann_learning_curves.png'), dpi=300, transparent=True)
plt.close()

# =============================================================================
# EVALUATION OF THE BEST CLASSIFIER OVER VALIDATION SET
# =============================================================================
# Load the best classifier and the validation set of the best fold 
best_model = load_model(os.path.join(classification_path, 'ann_best_classifier.h5'))
validate_set = file_management.load_lzma(os.path.join(classification_path, 'ann_best_fold_validation_set.lzma'))
# Evaluate the model over the validation set
y_pred = best_model.predict(X_train[validate_set])
# Round the predictions to the nearest integer
Y_pred = np.round(y_pred)
# Compute the confusion matrix
cm=confusion_matrix(y_train[validate_set],Y_pred)
# Compute the accuracy, recall, precision, f1-score and AUC 
accuracy = accuracy_score(y_train[validate_set], Y_pred)
recall = recall_score(y_train[validate_set], Y_pred)
precision = precision_score(y_train[validate_set], Y_pred)
f1score = f1_score(y_train[validate_set], Y_pred)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_train[validate_set], y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
# Save the results in a text file
with open(os.path.join(classification_path, 'best_model_validation_info.txt'), 'w') as f:
    f.write('------------------------------------------------------------------------ \n')
    f.write('Confussion matrix over validation set \n')
    f.write('------------------------------------------------------------------------ \n')
    f.write(f'{str(cm)} \n')
    f.write('------------------------------------------------------------------------ \n')
    f.write("Accuracy: %.2f%% \n" % (accuracy*100))
    f.write("Recall: %.2f%% \n" % (recall*100))
    f.write("Precision: %.2f%% \n" % (precision*100))
    f.write("F1-score: %.2f%% \n" % (f1score*100))
    f.write("AUC: %.2f \n" % (auc_keras))
    f.write("Proportion of positives: %.2f%% \n" % (100*sum(y_train[validate_set]/len(y_train[validate_set]))))
    f.write('------------------------------------------------------------------------ \n')
   
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
plt.savefig(os.path.join(classification_path, 'roc_curve_validation_ANN11.png'), dpi=300, transparent=True)
plt.close()

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
# Save the results in a text file 
with open(os.path.join(classification_path, 'best_model_testing_info.txt'), 'w') as f:
    f.write('------------------------------------------------------------------------ \n')
    f.write('Confussion matrix over test set \n')
    f.write('------------------------------------------------------------------------ \n')
    f.write(f'{str(cm)} \n')
    f.write('------------------------------------------------------------------------ \n')
    f.write("Accuracy: %.2f%% \n" % (accuracy*100))
    f.write("Recall: %.2f%% \n" % (recall*100))
    f.write("Precision: %.2f%% \n" % (precision*100))
    f.write("F1-score: %.2f%% \n" % (f1score*100))
    f.write("AUC: %.2f \n" % (auc_keras))
    f.write("Proportion of positives: %.2f%% \n" % (100*sum(y_test)/len(y_test)))
    f.write('------------------------------------------------------------------------ \n')
   
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
plt.savefig(os.path.join(classification_path, 'roc_curve_test_ANN11.png'), dpi=300, transparent=True)
plt.close()


# =============================================================================
# EVALUATION OF THE BEST CLASSIFIER OVER TREES IN THE TEST SET
# =============================================================================
# We evaluate the best classifier on the test set
y_test_pred = best_model.predict(X_test)

# given the prediction on each pixel that belongs to a single cluster_id, we compute the prediction on the cluster
# we do this by taking the mean of the predictions of the pixels that belong to the same cluster
# the cluster_id for each pixel is stored in the variable cluster_id_train and cluster_id_test

# we compute the prediction on the cluster for the test set
y_test_pred_cluster = np.zeros(len(set(cluster_id_test)))
y_test_label_cluster = np.zeros(len(set(cluster_id_test)))
for i, cluster_id in enumerate(set(cluster_id_test)):
    y_test_pred_cluster[i] = np.mean(y_test_pred[cluster_id_test==cluster_id])
    y_test_label_cluster[i] = np.mean(y_test[cluster_id_test==cluster_id])

y_test_pred = np.round(y_test_pred_cluster)
y_test = np.round(y_test_label_cluster)

# Compute the confusion matrix
cm=confusion_matrix(y_test,y_test_pred)
# Calculate the accuracy, recall, precision, f1-score and AUC
accuracy = accuracy_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
f1score = f1_score(y_test, y_test_pred)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_test_pred)
auc_keras = auc(fpr_keras, tpr_keras)
# Save the results in a text file 
with open(os.path.join(classification_path, 'best_model_testing_info_tree.txt'), 'w') as f:
    f.write('------------------------------------------------------------------------ \n')
    f.write('Confussion matrix over test set \n')
    f.write('------------------------------------------------------------------------ \n')
    f.write(f'{str(cm)} \n')
    f.write('------------------------------------------------------------------------ \n')
    f.write("Accuracy: %.2f%% \n" % (accuracy*100))
    f.write("Recall: %.2f%% \n" % (recall*100))
    f.write("Precision: %.2f%% \n" % (precision*100))
    f.write("F1-score: %.2f%% \n" % (f1score*100))
    f.write("AUC: %.2f \n" % (auc_keras))
    f.write("Proportion of positives: %.2f%% \n" % (100*sum(y_test)/len(y_test)))
    f.write('------------------------------------------------------------------------ \n')
   