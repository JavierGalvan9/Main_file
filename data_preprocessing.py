
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

parentDir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management


def groupby_split(df, labels='', inseparable_set='', test_size=0.05):
    # Given df and feature_2, groupby feature_2 and split each group into train and test sets
    # feature_1 is the feature to stratify the split
    # test_size is the size of the test set
    # Return a list of tuples (train, test) where train and test are dataframes
    
    # Create new dataframe with the first row for each unique value of feature_2
    df_new = df.groupby(inseparable_set).first().reset_index()
    # Stratify the split on feature_1
    train, test = train_test_split(df_new, test_size=test_size, stratify=df_new[labels], random_state=42)
    # From the original dataframe, select the rows that have the same values of feature_2 as the train and test dataframes
    train_df = df[df[inseparable_set].isin(train[inseparable_set])]
    test_df = df[df[inseparable_set].isin(test[inseparable_set])]
    # Return a list of tuples (train, test) where train and test are dataframes
    return train_df, test_df

# Define a function that, given a counter, returns the proportion of each class
def get_class_proportions(df, feature):
    counter = Counter(df[feature])
    total = sum(counter.values())
    new_dict = {cl: count / total for cl, count in counter.items()}
    # Sort the dictionary by key
    new_dict = {k: new_dict[k] for k in sorted(new_dict)}
    return new_dict

# Create a function that given a path creates the directory if it doesn't exist
# and returns the preprocessed data (X_train, X_test, y_train, y_test)
def data_preprocessing(path, n_samples=None, use_spectral_bands=True, use_indices=True):
    # Create a path to save/load the train and test sets which contains the features used
    if use_spectral_bands and not use_indices:
        train_test_path = os.path.join(path, 'Spectral bands')
    elif not use_spectral_bands and use_indices:
        train_test_path = os.path.join(path, 'Indices')
    elif use_spectral_bands and use_indices:
        train_test_path = os.path.join(path, 'Spectral bands and indices')
    else:
        # Print an error message and exit the program if the user doesn't choose any features
        print("Error: No features were selected!")
        sys.exit()
    # Check if the train and test sets directory exists and create it if it doesn't
    # This directory will be used to save the train and test sets
    # This is useful to avoid having to split the data again if the script is run again
    if not os.path.exists(train_test_path):
        os.makedirs(train_test_path)
        # Load data and remove unclassified pixels
        df = file_management.load_lzma('Processed Data/QPCR_labelled_df.lzma')
        df.dropna(inplace=True)
        # Sample a subset of the data
        if n_samples is not None:
            df = df.sample(n=n_samples)
        # df.loc[df['PCR'] ==np.nan, 'PCR'] = 0

        # Split the data into train and test sets (stratified to keep the same proportion of labels in each set but 
        # keeping each unique tree cluster in only one set)
        train_df, test_df = groupby_split(df, labels='PCR', inseparable_set='cluster_id', test_size=0.1)
        # Get the cluster ids of the train and test sets
        cluster_id_train = train_df['cluster_id']
        cluster_id_test = test_df['cluster_id']
        # Get the latitudes and longitudes of the training and testing sets
        lat_train = train_df['Lats']
        lon_train = train_df['Longs']
        lat_test = test_df['Lats']
        lon_test = test_df['Longs']

        # Plot the trees used for training and testing
        original_df = file_management.load_lzma('Processed Data/dataset.lzma')
        metadata_df = file_management.load_lzma('Processed Data/metadata_df.lzma')
        pan_shape = metadata_df['original_shape']
        # Loc three columns of original dataset
        original_df = original_df.loc[:, ['N', 'R', 'G', 'Longs', 'Lats']]	
        rgb_df = (255*original_df/original_df.max()).astype(np.uint8)
        rgb = np.dstack((rgb_df['N'].values.reshape(pan_shape),rgb_df['R'].values.reshape(pan_shape),rgb_df['G'].values.reshape(pan_shape)))
        fig = plt.figure()
        plt.imshow(rgb,
                extent=[original_df['Longs'].min(), original_df['Longs'].max(), original_df['Lats'].min(), original_df['Lats'].max()])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.ticklabel_format(useOffset=False)
        # Draw a circle around the trees used for training and testing 
        plt.scatter(lon_train, lat_train, s=(72/300)**2, alpha=1, c='yellow', marker='s')
        plt.scatter(lon_test, lat_test, s=(72/300)**2, alpha=1, c='indigo', marker='s')
        plt.legend(['Train', 'Test'])
        # Save the plot
        path = 'Images/Train_test_layout'
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, 'layout_rgb.png'), dpi=300, transparent=True)
        plt.close()

        # Split the data into train and test dataframes (stratified to keep the same proportion of classes in each set)
        # train_df, test_df = train_test_split(df, test_size=0.05, stratify=df['PCR'], random_state=88)
        # Choose the features to use in the classification task
        if use_spectral_bands and not use_indices:
            spectral_bands = ['C', 'B', 'G', 'Y', 'R', 'RE', 'N', 'N2']
            X_train = train_df.loc[:, spectral_bands] # only spectral bands
            X_test = test_df.loc[:, spectral_bands] # only spectral bands
        elif not use_spectral_bands and use_indices:
            X_train = train_df.iloc[:, 8:-4] # indices
            X_test = test_df.iloc[:, 8:-4] # indices            
        elif use_spectral_bands and use_indices:
            spectral_bands = ['C', 'B', 'G', 'Y', 'R', 'RE', 'N', 'N2']
            X_train = train_df.loc[:, spectral_bands + list(train_df.columns[8:-4])]
            X_test = test_df.loc[:, spectral_bands + list(test_df.columns[8:-4])]
        else:
            # Print an error message and exit the program if the user doesn't choose any features
            print("Error: No features were selected!")
            sys.exit()

        # Print the column names used in X_train and X_test
        print("Features used in the classification task: ")
        print(X_train.columns)

        # Save the labels in a separate variable 
        y_train = train_df['PCR'].values
        y_test = test_df['PCR'].values
        
        # Print the number of samples in each set
        print('Training set size: ', len(train_df))
        print('Test set size: ', len(test_df))

        # Ensure that the proportion of samples in each set is the same as the original dataset
        # Print the proportion samples in each set by dividing the counter by the total number of samples
        feature = 'PCR'
        print('Original dataset proportions: ', get_class_proportions(df, feature))
        print('Training set proportions: ', get_class_proportions(train_df, feature) )
        print('Test set proportions: ', get_class_proportions(test_df, feature) )

        # Print the number of unique cluster_id in each set
        print('Number of unique cluster_id in the original dataset: ', len(df['cluster_id'].unique()))
        print('Number of unique cluster_id in the training set: ', len(train_df['cluster_id'].unique()))
        print('Number of unique cluster_id in the test set: ', len(test_df['cluster_id'].unique()))

        # Data preprocessing - Standard normalization of the features 
        # This is done to avoid the features with higher values to dominate the training process
        # The mean and standard deviation of the train set are used to normalize the test set 
        std_scale = StandardScaler()
        # Fit the scaler to the train set and transform it
        X_train = std_scale.fit_transform(X_train)
        # Transform the test set using the same mean and standard deviation
        X_test = std_scale.transform(X_test)
        # Save the train and test sets
        file_management.save_lzma(X_train, os.path.join(train_test_path, 'X_train.lzma'), '')
        file_management.save_lzma(y_train, os.path.join(train_test_path, 'Y_train.lzma'), '')
        file_management.save_lzma(X_test, os.path.join(train_test_path, 'X_test.lzma'), '')
        file_management.save_lzma(y_test, os.path.join(train_test_path, 'Y_test.lzma'), '')
        # Save the scaler to use it later to normalize new data
        file_management.save_lzma(std_scale, os.path.join(train_test_path, 'scaler.lzma'), '')
        # Save the cluster ids of the train and test sets
        file_management.save_lzma(cluster_id_train, os.path.join(train_test_path, 'cluster_id_train.lzma'), '')
        file_management.save_lzma(cluster_id_test, os.path.join(train_test_path, 'cluster_id_test.lzma'), '')

        return X_train, X_test, y_train, y_test, std_scale, cluster_id_train, cluster_id_test
    else:
        # Load the train and test sets
        X_train = file_management.load_lzma(os.path.join(train_test_path, 'X_train.lzma'))
        y_train = file_management.load_lzma(os.path.join(train_test_path, 'Y_train.lzma'))
        X_test = file_management.load_lzma(os.path.join(train_test_path, 'X_test.lzma'))
        y_test = file_management.load_lzma(os.path.join(train_test_path, 'Y_test.lzma'))
        # Load the scaler
        std_scale = file_management.load_lzma(os.path.join(train_test_path, 'scaler.lzma'))
        # Load the cluster ids of the train and test sets
        cluster_id_test = file_management.load_lzma(os.path.join(train_test_path, 'cluster_id_test.lzma'))
        cluster_id_train = file_management.load_lzma(os.path.join(train_test_path, 'cluster_id_train.lzma'))

        return X_train, X_test, y_train, y_test, std_scale, cluster_id_train, cluster_id_test