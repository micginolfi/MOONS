#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 

@author: mginolfi

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import tensorflow as tf
import scipy.ndimage
from sklearn.model_selection import train_test_split


#%%
""""""""""""""""""""""""""""""""""""""""""""
""" Dataset preparation (run once)       """
""""""""""""""""""""""""""""""""""""""""""""


""" load the combined dataset pickle file and split in train - validation - test. save local copies """ 

# Read the combined dataset
reconstructed_df = pd.read_pickle('final_combined_dataset.pickle')

# show keys
reconstructed_df.columns

# First Split: Train (including validation) and Test
df_train_val, df_test = train_test_split(reconstructed_df, test_size=0.15, random_state=42) # 15% test

# Second Split: Train and Validation from X_temp and Y_temp
# Note: 15% of the remaining 85% is 0.1765 (approximately 17.65%)
df_train, df_val = train_test_split(df_train_val, test_size=0.1765, random_state=42) # About 15% of total

# Saving the Split DataFrames
df_train.to_pickle('train_dataset.pickle')
df_val.to_pickle('validation_dataset.pickle')
df_test.to_pickle('test_dataset.pickle')

#%%

""""""""""""""""""""""""""""""""
""" Loading & process data   """
""""""""""""""""""""""""""""""""

#%%
""" load training set, make X & Y dataset and normalise"""

df_train = pd.read_pickle('train_dataset.pickle')

# Extract features for training set
all_spectra_train = np.stack(df_train['combined_spectrum'].values)
all_skyFlux_train = np.stack(df_train['combined_skyFlux'].values)

# Normalization of training spectra
max_value_train = all_spectra_train.max()
all_spectra_train_normalized = all_spectra_train / max_value_train

# make X train
X_train = np.stack((all_spectra_train_normalized, all_skyFlux_train), axis=-1)

# Extract labels for training set
all_redshift_train = df_train['z'].values
all_stellar_masses_train = df_train['log_m'].values
all_sfr_train = np.log10(df_train['sfr'].values)

# define labels: make Y
Y_train = np.column_stack((all_redshift_train, all_stellar_masses_train, all_sfr_train))

# Calculate mean and standard deviation for each label type in the training set
Y_train_mean = Y_train.mean(axis=0)
Y_train_std = Y_train.std(axis=0)

# Normalize training labels
Y_train_normalized = (Y_train - Y_train_mean) / Y_train_std

del df_train
del all_spectra_train
del all_skyFlux_train

#%%
""" load validation set, make X & Y dataset and normalise"""

df_val = pd.read_pickle('validation_dataset.pickle')

# Extract features for validation set
all_spectra_val = np.stack(df_val['combined_spectrum'].values)
all_skyFlux_val = np.stack(df_val['combined_skyFlux'].values)

# Normalization of validation spectra
all_spectra_val_normalized = all_spectra_val / max_value_train

# make X val
X_val = np.stack((all_spectra_val_normalized, all_skyFlux_val), axis=-1)

# Extract labels for validation set
all_redshift_val = df_val['z'].values
all_stellar_masses_val = df_val['log_m'].values
all_sfr_val = np.log10(df_val['sfr'].values)

# define labels
Y_val = np.column_stack((all_redshift_val, all_stellar_masses_val, all_sfr_val))

# Normalize validation labels
Y_val_normalized = (Y_val - Y_train_mean) / Y_train_std

del df_val
del all_spectra_val
del all_skyFlux_val

#%%
""" load test set, make X & Y dataset and normalise"""

df_test = pd.read_pickle('test_dataset.pickle')

# Extract features for test set
all_spectra_test = np.stack(df_test['combined_spectrum'].values)
all_skyFlux_test = np.stack(df_test['combined_skyFlux'].values)

# Normalization of validation spectra
all_spectra_test_normalized = all_spectra_test / max_value_train

# make X test
X_test = np.stack((all_spectra_test_normalized, all_skyFlux_test), axis=-1)

# Extract labels for test set
all_redshift_test = df_test['z'].values
all_stellar_masses_test = df_test['log_m'].values
all_sfr_test = np.log10(df_test['sfr'].values)

# define labels
Y_test = np.column_stack((all_redshift_test, all_stellar_masses_test, all_sfr_test))

# Normalize test labels
Y_test_normalized = (Y_test - Y_train_mean) / Y_train_std

#%%

""""""""""""""""""""""""""""""""
"""   Modelling              """
""""""""""""""""""""""""""""""""

#%%
""" Create model """

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, BatchNormalization
import tensorflow as tf

def create_model(input_shape):
    
    # print(input_shape)
    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(2, (1, 2),  strides=(1, 1), activation='elu')(inputs)
    # x = BatchNormalization()(x)  # Batch Normalization after convolution
    x = Conv2D(1, (100, 1), strides=(4, 1), activation='elu')(x)
    x = Conv2D(1, (10, 1), strides=(4, 1), activation='elu')(x)
    x = Flatten()(x)
    
    # First dense hidden layer
    # x = Dense(32, activation='relu')(inputs)
    
    # First dense hidden layer
    x = Dense(128, activation='elu')(x)
    # x = BatchNormalization()(x)  # Batch Normalization after dense layer
    x = Dropout(0.2)(x)  # Dropout layer

    # Second dense hidden layer
    x = Dense(64, activation='elu')(x)
    # x = BatchNormalization()(x)  # Batch Normalization after dense layer
    x = Dropout(0.2)(x)  # Dropout layer

    # Third dense hidden layer
    x = Dense(32, activation='elu')(x)
    # x = BatchNormalization()(x)  # Batch Normalization after dense layer
    x = Dropout(0.2)(x)  # Dropout layer
    
    # Task-specific layers
    # Redshift prediction
    redshift_output = Dense(1, activation='linear', name='redshift')(x)
    
    # Stellar mass prediction
    stellar_mass_output = Dense(1, activation='linear', name='stellar_mass')(x)
    
    # Star formation rate prediction
    sfr_output = Dense(1, activation='linear', name='sfr')(x)
    
    # Define model
    model = Model(inputs=inputs, outputs=[redshift_output, stellar_mass_output, sfr_output])
    
    return model


model = create_model(np.expand_dims(X_train[0], -1).shape) # np.expand_dims(X_train[0], -1).shape = (12217, 2, 1)

model.summary()

#%%
""" Class for metrics tracking """
from tensorflow.keras.callbacks import Callback

class MetricsPlotter(Callback):
    def __init__(self, task_names):
        # Initialize the lists to store the metrics
        self.train_loss = []
        self.val_loss = []
        self.task_metrics = {task: {'train_mae': [], 'val_mae': []} for task in task_names}

    def on_epoch_end(self, epoch, logs=None):
        # Append the losses
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        # Append the task-specific metrics
        for task in self.task_metrics.keys():
            self.task_metrics[task]['train_mae'].append(logs.get(f'{task}_mae'))
            self.task_metrics[task]['val_mae'].append(logs.get(f'val_{task}_mae'))

        # Plot the metrics
        self.plot_metrics(epoch)

    def plot_metrics(self, epoch):
        # Clear the previous plot
        plt.clf()
    
        # Determine the number of rows and columns for subplots
        num_tasks = len(self.task_metrics)
        num_plots = num_tasks + 1  # +1 for the total loss plot
        cols = 2  # number of columns as prederred
        rows = (num_plots + cols - 1) // cols  # Calculate rows needed
    
        # Create subplots
        fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    
        # Plot total training and validation loss
        axs[0, 0].plot(range(epoch+1), self.train_loss, label='Training Loss')
        axs[0, 0].plot(range(epoch+1), self.val_loss, label='Validation Loss')
        axs[0, 0].set_title('Total Loss')
        axs[0, 0].legend()
    
        # Plot task-specific metrics
        plot_index = 1  # Start from the second plot
        for task, metrics in self.task_metrics.items():
            ax = axs[plot_index // cols, plot_index % cols]
            ax.plot(range(epoch+1), metrics['train_mae'], label=f'{task} Training MAE')
            ax.plot(range(epoch+1), metrics['val_mae'], label=f'{task} Validation MAE')
            ax.set_title(f'{task.capitalize()} MAE')
            ax.legend()
            plot_index += 1
    
        # Adjust layout
        plt.tight_layout()
        plt.show()
        
#%%
""" Compile & run  model """

model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'redshift': 'mse', 
                    'stellar_mass': 'mse', 
                    'sfr': 'mse'},
              loss_weights={'redshift': 2.0, 'stellar_mass': 1.0, 'sfr': 1.0},
              metrics={'redshift': 'mae', 'stellar_mass': 'mae', 'sfr': 'mae'})

# Early Stopping Callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_redshift_loss',  # Monitor the validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To log when training is stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
)


# Instantiate the callback with the task names
task_names = ['redshift', 'stellar_mass', 'sfr']
metrics_plotter = MetricsPlotter(task_names=task_names)


# Train model with validation data, and add callback with visualisation
history = model.fit(X_train, {'redshift': Y_train_normalized[:, 0], 'stellar_mass': Y_train_normalized[:, 1], 'sfr': Y_train_normalized[:, 2]},
                    validation_data=(X_val, {'redshift': Y_val_normalized[:, 0], 'stellar_mass': Y_val_normalized[:, 1], 'sfr': Y_val_normalized[:, 2]}),
                    shuffle=True,
                    epochs=400,
                    batch_size=1024,
                    callbacks=[early_stopping, metrics_plotter])

#%%
""" Check history """

def plot_history(history, task, early_stopping_epoch=None):
    
    plt.figure(figsize=(12, 4))

        
    if task == 'total':
        # Plot total training & validation loss values
        plt.subplot(1, 1, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Total Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        if early_stopping_epoch is not None: plt.axvline(x=early_stopping_epoch, color='gray', linestyle='--')
        
    else:
        # Plot training & validation loss values for specific task
        plt.subplot(1, 2, 1)
        plt.plot(history.history[task+'_loss'])
        plt.plot(history.history['val_'+task+'_loss'])
        plt.title('Model Loss for ' + task.capitalize())
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        if early_stopping_epoch is not None: plt.axvline(x=early_stopping_epoch, color='gray', linestyle='--')


        # Plot training & validation MAE values for specific task
        plt.subplot(1, 2, 2)
        plt.plot(history.history[task+'_mae'])
        plt.plot(history.history['val_'+task+'_mae'])
        plt.title('Model MAE for ' + task.capitalize())
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        if early_stopping_epoch is not None: plt.axvline(x=early_stopping_epoch, color='gray', linestyle='--')



    plt.show()


# #Plot total loss and tasks metrics 
plot_history(history, 'total', early_stopping_epoch=np.argmin(history.history['val_redshift_loss']))
plot_history(history, 'redshift', early_stopping_epoch=np.argmin(history.history['val_redshift_loss']))
plot_history(history, 'stellar_mass', early_stopping_epoch=np.argmin(history.history['val_redshift_loss']))
plot_history(history, 'sfr', early_stopping_epoch=np.argmin(history.history['val_redshift_loss']))

#%%
""" Check global predictions on test-set """

test_metrics = model.evaluate(X_test, {'redshift': Y_test_normalized[:, 0], 'stellar_mass': Y_test_normalized[:, 1], 'sfr': Y_test_normalized[:, 2]})

# compute prediction
predictions = np.array(model.predict(X_test))

# Reshape predictions to remove the extra dimension and match Y_test
predictions_reshaped = predictions.squeeze()  


def inverse_transform(normalized_values, means, stds):
    return normalized_values * stds + means

# Applying the inverse transformation to predictions
predictions_rescaled = np.array([inverse_transform(predictions[i], Y_train_mean[i], Y_train_std[i]) for i in range(predictions.shape[0])])

def plot_predictions(predicted, actual, task_name):
    plt.scatter(actual, predicted, alpha=0.1)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values for {task_name}')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
    plt.show()

# Plotting predictions vs actual values for each task
for i, task_name in enumerate(['Redshift', 'Stellar Mass', 'SFR']):
    plot_predictions(predictions_rescaled[i], Y_test[:, i], task_name)
