
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, GlobalAveragePooling2D)
from tensorflow.keras.models import Model, Sequential

# Set random seeds for TensorFlow and NumPy
tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# ANN: 1 pixel = 1 tree
# =============================================================================
# Define the model architecture
def NN_model(n_features, dropout_rate=0.2):
    # Define the model architecture and hyperparameters
    classifier = Sequential()
    # Input layer
    classifier.add(Dense(units = 128, input_dim = n_features, kernel_initializer = 'glorot_normal'))
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch
    classifier.add(Activation('relu')) # Activation function
    classifier.add(Dropout(dropout_rate)) # Dropout layer to avoid overfitting
    # Second layer
    classifier.add(Dense(units = 64, kernel_initializer = 'glorot_normal')) # Dense layer
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch
    classifier.add(Activation('relu')) # Activation function
    #classifier.add(Dropout(rate = 0.2))
    # Third layer
    classifier.add(Dense(units = 32, kernel_initializer = 'glorot_normal')) # Dense layer
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch
    classifier.add(Activation('relu')) # Activation function
    #classifier.add(Dropout(rate = 0.2))
    # Fourth layer
    classifier.add(Dense(units = 8, kernel_initializer = 'glorot_normal')) # Dense layer
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch
    classifier.add(Activation('relu')) # Activation function
    # Output layer (binary classification)
    # add a dense layer with a single neuron and sigmoid activation function with the glorot normal initializer
    # the output of the sigmoid function is the probability of the sample belonging to class 1
    # the probability of the sample belonging to class 0 is 1 - probability of the sample belonging to class 1
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_normal', activation = "sigmoid"))  
      
    # Compile the model with the Adam optimizer and the binary cross-entropy loss function
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", 'AUC'])
    return classifier

# Define the model architecture
def customizable_NN_model(n_features, input_dim=None, dropout_rate=0.2, n_layers=4, hidden_units=32, activation='relu'):
    # Define the model architecture and hyperparameters 
    classifier = Sequential()
    classifier.add(Dense(units=hidden_units, activation=activation, input_dim=input_dim, kernel_initializer = 'glorot_normal'))
    # Add dropout layer to avoid overfitting 
    classifier.add(Dropout(dropout_rate))
    for i in range(n_layers-1):
        classifier.add(Dense(units=hidden_units, activation=activation, kernel_initializer = 'glorot_normal'))
        classifier.add(BatchNormalization())         
    # Output layer (binary classification) 
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_normal',  activation = "sigmoid"))
    # Compile the model with the Adam optimizer and the binary cross-entropy loss function
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", 'AUC'])
    return classifier

# Define the model architecture and hyperparameters for the L2-regularized model 
def L2_regularized_model(n_features, dropout_rate=0.2):
    # Define the model architecture and hyperparameters 
    classifier = Sequential()
    # Input layer with L2 regularization 
    classifier.add(Dense(units = 128, input_dim = n_features, kernel_initializer = 'glorot_normal',  activation = "relu",
                         kernel_regularizer=regularizers.L2(0.01)))
    # Add dropout layer to avoid overfitting 
    classifier.add(Dropout(dropout_rate))
    # Second layer
    classifier.add(Dense(units = 64, kernel_initializer = 'glorot_normal',  activation = "relu", 
                         kernel_regularizer=regularizers.L2(0.01)))
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch
    #classifier.add(Dropout(rate = 0.2))
    # Third layer
    classifier.add(Dense(units = 32, kernel_initializer = 'glorot_normal',  activation = "relu",
                         kernel_regularizer=regularizers.L2(0.01)))
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch
    #classifier.add(Dropout(rate = 0.2))
    # Fourth layer
    classifier.add(Dense(units = 8, kernel_initializer = 'glorot_normal',  activation = "relu",
                         kernel_regularizer=regularizers.L2(0.01)))
    classifier.add(BatchNormalization()) # Normalize the activations of the previous layer at each batch                         
    # Output layer (binary classification) 
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_normal',  activation = "sigmoid"))

    # Compile the model with the Adam optimizer and the binary cross-entropy loss function
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", 'AUC'])
    return classifier

# Inception implementation
def inceptionV3_model(input_shape=(224, 224, 3)):
        # Load the pre-trained VGG16 model with weights trained on ImageNet
    base_model = tf.keras.applications.VGG16(input_shape=input_shape,
                                            include_top=False,
                                            weights='imagenet')

    # Freeze the layers in the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom classifier on top for binary segmentation
    x = base_model.output
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

# Vgg16 implementation
def VGG16_model(input_shape=(224, 224, 3)):
    # Load the VGG16 model with ImageNet weights, excluding the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add a global average pooling layer and a dense layer with a sigmoid activation function
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)

    # Create a new model with the modified output layer
    model = Model(inputs=base_model.input, outputs=x)

    # Freeze the layers of the base model so that only the new dense layer is trainable
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model with binary cross-entropy loss and the Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, Dense, Input, UpSampling2D


def ResNet50_model(input_shape):
    # Load pre-trained ResNet50 model with imagenet weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Create a new model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model