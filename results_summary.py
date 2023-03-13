# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 12:41:04 2022

@author: UX325
"""

import glob
import os

import pandas as pd

# ----------------------------------------------------------------
# Get the best model's performance on validation and testing sets
# ----------------------------------------------------------------
performance_metrics = ["Accuracy", "Recall", "Precision", "F1-score", "AUC"]
df = pd.DataFrame(columns=performance_metrics)
for item in ['validation', 'testing']:
    path = 'Classification results/*/best_model_'+item+'_info.txt'
    for model_path in glob.glob(path):
        with open(model_path) as f:
            contents = f.read()
        ## Split the string into a list of words
        x = contents.split()
        row_idx = df.shape[0]
        ## Get the model name from the path and add it to the dataframe
        model_path = os.path.normpath(model_path)
        model_name = model_path.split(os.sep)[-2]
        df.loc[model_name] = [None, None, None, None, None]
        ## Get the performance metrics from the list of words and add them to the dataframe
        for metric in performance_metrics:
            metric_idx = x.index(metric+":") 
            df.loc[model_name, metric] = x[metric_idx+1]
    # Sort the dataframe by F1-score
    df = df.sort_values('F1-score', ascending=False)
    # Save the dataframe to an excel file
    parent_dir = 'Classification results'
    df.to_excel(os.path.join(parent_dir, item+"_comparison.xlsx"))  
    