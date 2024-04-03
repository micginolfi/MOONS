#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 

@author: mginolfi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D,AveragePooling1D, Input, concatenate, BatchNormalization, Activation, Multiply, Permute, BatchNormalization, Attention
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import callbacks
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import tensorflow as tf
import keras
from keras import backend as k
from keras.callbacks import Callback, LearningRateScheduler

#%%
""" utils to make emission & absorption lines masks""" 

# Lines rest wavelengths are in AA
# I took Vacuum values (https://classic.sdss.org/dr7/products/spectra/vacwavelength.php)
# conversion here: https://www.sdss3.org/dr8/spectro/spectra.php
# update 02Apr2024: adding more lines from Filippo

emission_lines_rest = {
    'OII_1': 3727.1,            # Oxygen II 
    'OII_2': 3729.9,            # Oxygen II
    'H_beta': 4862.7,           # Hydrogen-beta
    'OIII_1': 4960.3,           # Oxygen III
    'OIII_2': 5008.2,           # Oxygen III
    'NII_1': 6549.8,            # Nitrogen II
    'H_alpha': 6564.6,          # Hydrogen-alpha
    'NII_2': 6585.3,            # Nitrogen II
    'SII_1': 6718.3,            # Sulfur II
    'SII_2': 6732.7,            # Sulfur II
    'SIII_1': 9070,             # Sulfur III
    'SIII_2': 9532,             # Sulfur III
    'Lyalpha': 1206,            # Lyman-alpha
    'NV': 1240,                 # Nitrogen V
    'SiII_1': 1260,             # Silicon II
    'OI': 1303,                 # Oxygen I
    'CII': 1334,                # Carbon II
    'SiIV_1': 1393,             # Silicon IV
    'SiIV_2': 1402,             # Silicon IV
    'SiII_2': 1526,             # Silicon II
    'CIV_1': 1548,              # Carbon IV
    'HeII_1': 1640,             # Helium II
    'OII_3': 1660,              # Oxygen II
    'OII_4': 1666,              # Oxygen II
    'CIII': 1909,               # Carbon III
    'CN': 3875,                 # Cyanide radical
    'CaII_1': 3933,             # Calcium II
    'CaII_2': 3969,             # Calcium II
    'FeII': 4668,               # Iron II
    'MgI_1': 5167,              # Magnesium I
    'MgI_2': 5172,              # Magnesium I
    'MgI_3': 5183,              # Magnesium I
    'FeI': 5270,                # Iron I
    'NaI_1': 5892,              # Sodium I
    'TiO_1': 6180,              # Titanium Oxide
    'TiO_2': 7150,              # Titanium Oxide
    'NaI_2': 8183,              # Sodium I
    'NaI_3': 8195,              # Sodium I
    'CaII_3': 8489,             # Calcium II
    'CaII_4': 8542,             # Calcium II
    'CaII_5': 8662,             # Calcium II
    'HeII_2': 10830             # Helium II
}



# read the wavelength axis 
import pickle
with open('vacuum_wavelength.pkl', 'rb') as file:
    wavelength_axis = pickle.load(file)

def create_emission_output(z, wavelength_axis, emission_lines_rest, threshold=1.0, adjacent_channels=5):
    """
    Create an emission line mask output array based on the redshift (z), wavelength axis, emission lines,
    and a threshold value (in Angstrom) to check if the line is present in the wavelength domain.
    The number of adjacent channels to mark on either side of the predicted position is determined by
    the adjacent_channels parameter.
    """
    emission_output = np.zeros_like(wavelength_axis)
    for line in emission_lines_rest:
        emitted_wavelength = emission_lines_rest[line]
        observed_wavelength = emitted_wavelength * (1 + z)
        idx, nearest_wavelength = find_nearest_index(wavelength_axis, observed_wavelength)

        # Check if the nearest wavelength is within the threshold
        if np.abs(nearest_wavelength - observed_wavelength) <= threshold:
            # Mark the predicted position and adjacent channels on either side
            start_idx = max(idx - adjacent_channels, 0)  # Ensure index is not less than 0
            end_idx = min(idx + adjacent_channels + 1, len(emission_output))  # Ensure index does not exceed array length
            emission_output[start_idx:end_idx] = 1 
    return emission_output

def find_nearest_index(array, value):
    """ Find nearest index in array for given value and return the index and the nearest value """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# visualisation: check the emission line mask for a given redshift
z = 1 # Example redshift
emission_output_example = create_emission_output(z, wavelength_axis, emission_lines_rest)
plt.plot(wavelength_axis, emission_output_example, lw=1)
plt.xlabel('wavelengt [AA]')

#%%
""" Inputs for redshift domain binning """
import scipy.ndimage

# Step resolution for redshift bins
step_resolution = 0.003
bin_categories = np.arange(0.6, 3, step=step_resolution).round(decimals=6)

def create_categorical_labels(labels, sigma):
    categorical_labels = np.zeros((len(labels), len(bin_categories)))
    for i, label in enumerate(labels):
        bin_index = np.digitize(label, bin_categories) - 1  # -1 for zero-indexing
        categorical_labels[i, bin_index] = 1
        categorical_labels[i, :] = scipy.ndimage.gaussian_filter(categorical_labels[i, :], sigma=sigma)
    return categorical_labels

#%%
""" open sky flux array """

# read sky flux
import pickle
with open('sky_flux.pkl', 'rb') as file:
    all_sky = pickle.load(file)
    
mask_sky = np.log10(all_sky)>-4.8

plt.plot(wavelength_axis, all_sky)
plt.plot(wavelength_axis[mask_sky], all_sky[mask_sky])
plt.plot(wavelength_axis[mask_sky], all_sky[mask_sky])
plt.semilogy()


#%%
""" load training set, make X & Y dataset and normalise"""
""" train + val mixed -- val later taken from test """

# Load the training and validation dataframes
df_temp1 = pd.read_pickle('train_dataset.pickle')
df_temp2 = pd.read_pickle('validation_dataset.pickle')

# Concatenate the dataframes
df_train = pd.concat([df_temp1, df_temp2], ignore_index=True)

# Load and concatenate features (spectra)
with open('noise_all_spectra_train_normalized.pkl', 'rb') as file:
    all_spectra_temp1_normalized = pickle.load(file)
with open('noise_all_spectra_val_normalized.pkl', 'rb') as file:
    all_spectra_temp2_normalized = pickle.load(file)
    
all_spectra_train_normalized = np.concatenate((all_spectra_temp1_normalized, all_spectra_temp2_normalized))
all_spectra_train_normalized[:, mask_sky] = 0
X_train = all_spectra_train_normalized

# Extract labels for training set
all_redshift_train = df_train['z'].values

# define labels: make Y
Y_train = all_redshift_train

emission_line_masks_train = np.array([create_emission_output(z, wavelength_axis, emission_lines_rest) for z in all_redshift_train])

# Convert labels to categorical
Y_train_categorical = create_categorical_labels(Y_train, sigma=0)


# Load and concatenate continuum data
with open('noise_all_spectra_train_continuum.pkl', 'rb') as file:
    temp1_continuum = pickle.load(file)
with open('noise_all_spectra_val_continuum.pkl', 'rb') as file:
    temp2_continuum = pickle.load(file)
train_continuum = np.concatenate((temp1_continuum, temp2_continuum))


Y_train_mstar = df_train['log_m'].values
Y_train_sfr = np.log10(df_train['sfr'].values)


# Clean up to free memory
del df_train

#%%
""" load validation set, make X & Y dataset and normalise"""

df_val = pd.read_pickle('test_dataset.pickle')

# mask_restrict = df_val['MAG_vista-H'] < 24
# df_val = df_val[mask_restrict]

# Extract features for validation set
import pickle
with open('noise_all_spectra_test_normalized.pkl', 'rb') as file:
    all_spectra_val_normalized = pickle.load(file)

# mic
all_spectra_val_normalized[:, mask_sky] = 0

# all_spectra_val_normalized = all_spectra_val_normalized[mask_restrict]

X_val = all_spectra_val_normalized

# Extract labels for validation set
all_redshift_val = df_val['z'].values

# define labels
Y_val = all_redshift_val
emission_line_masks_val = np.array([create_emission_output(z, wavelength_axis, emission_lines_rest) for z in all_redshift_val])

Y_val_categorical = create_categorical_labels(Y_val, sigma=0)


# -------- mstar - sfr tasks 
with open('noise_all_spectra_test_continuum.pkl', 'rb') as file:
    val_continuum = pickle.load(file)

Y_val_mstar = df_val['log_m'].values
Y_val_sfr = np.log10(df_val['sfr'].values)

del df_val


#%%
""" load test set, make X & Y dataset and normalise"""

df_test = pd.read_pickle('test_dataset.pickle')

# mask_restrict = df_test['MAG_vista-H'] < 24
# df_test = df_test[mask_restrict]

df_test.columns

# Extract features for test set
import pickle
with open('noise_all_spectra_test_normalized.pkl', 'rb') as file:
    all_spectra_test_normalized = pickle.load(file)
    
# mic
all_spectra_test_normalized[:, mask_sky] = 0

# all_spectra_test_normalized = all_spectra_test_normalized[mask_restrict]

X_test = all_spectra_test_normalized

# Extract labels for test set
all_redshift_test = df_test['z'].values

# define labels
Y_test = all_redshift_test
emission_line_masks_test = np.array([create_emission_output(z, wavelength_axis, emission_lines_rest) for z in all_redshift_test])

Y_test_categorical = create_categorical_labels(Y_test, sigma=0)

# -------- mstar - sfr tasks
with open('noise_all_spectra_test_continuum.pkl', 'rb') as file:
    test_continuum = pickle.load(file)

Y_test_mstar = df_test['log_m'].values
Y_test_sfr = np.log10(df_test['sfr'].values)



#%%
""""""""""""""""""""""""""""""""""""""""""""
"""   Normalize mstar / sfr and continuum """
""""""""""""""""""""""""""""""""""""""""""""
# Calculating the mean and standard deviation of the training continuum data
continuum_mean = train_continuum.mean()
continuum_std = train_continuum.std()

# Normalizing the continuum data
train_continuum = (train_continuum - continuum_mean) / continuum_std
val_continuum = (val_continuum - continuum_mean) / continuum_std
test_continuum = (test_continuum - continuum_mean) / continuum_std



mstar_mean, mstar_std = Y_train_mstar.mean(), Y_train_mstar.std()
sfr_mean, sfr_std = Y_train_sfr.mean(), Y_train_sfr.std()

Y_train_mstar = (Y_train_mstar - mstar_mean) / mstar_std
Y_train_sfr = (Y_train_sfr - sfr_mean) / sfr_std

# Apply the same transformation to validation and test sets
Y_val_mstar = (Y_val_mstar - mstar_mean) / mstar_std
Y_val_sfr = (Y_val_sfr - sfr_mean) / sfr_std

Y_test_mstar = (Y_test_mstar - mstar_mean) / mstar_std
Y_test_sfr = (Y_test_sfr - sfr_mean) / sfr_std


#%%
""""""""""""""""""""""""""""""""
"""   Modelling              """
""""""""""""""""""""""""""""""""

from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, Concatenate
from keras.models import Model

def create_multitask_model(input_shape, num_categories, continuum_shape):
    
    
    # Shared spectral layers
    spectral_input = Input(shape=input_shape)
    
    x = Conv1D(filters=16, kernel_size=11, activation='relu')(spectral_input)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(filters=32, kernel_size=11, activation='relu')(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(filters=64, kernel_size=11, activation='relu')(x)
    x = MaxPooling1D(pool_size=5)(x)

    shared_spectral_features = Flatten()(x)

    # Emission line location prediction branch
    emission_branch = Dense(64, activation='relu')(shared_spectral_features)
    emission_branch = Dropout(0.2)(emission_branch)
    emission_branch = Dense(128, activation='relu')(emission_branch)
    emission_branch = Dropout(0.2)(emission_branch)
    emission_output = Dense(input_shape[0], activation='sigmoid', name='emission_output')(emission_branch)

    # Continuum input and processing
    continuum_input = Input(shape=continuum_shape)
    continuum_processed = Dense(32, activation='relu')(continuum_input)

    # Combine spectral features with processed continuum
    combined_features = Concatenate()([shared_spectral_features, continuum_processed, emission_branch])

    # Redshift prediction branch
    redshift_branch = Dense(64, activation='relu')(combined_features)
    redshift_branch = Dropout(0.2)(redshift_branch)
    redshift_branch = Dense(128, activation='relu')(redshift_branch)
    redshift_branch = Dropout(0.2)(redshift_branch)    
    redshift_output = Dense(num_categories, activation='softmax', name='redshift_output')(redshift_branch)

    # Stellar Mass (Mstar) prediction branch
    mstar_branch = Dense(64, activation='relu')(combined_features)
    mstar_branch = Dropout(0.2)(mstar_branch)
    mstar_branch = Dense(32, activation='relu')(mstar_branch)
    mstar_branch = Dropout(0.2)(mstar_branch)
    mstar_output = Dense(1, activation='linear', name='mstar_output')(mstar_branch)

    # Star Formation Rate (SFR) prediction branch
    sfr_branch = Dense(64, activation='relu')(combined_features)
    sfr_branch = Dropout(0.2)(sfr_branch)
    sfr_branch = Dense(32, activation='relu')(sfr_branch)
    sfr_branch = Dropout(0.2)(sfr_branch)
    sfr_output = Dense(1, activation='linear', name='sfr_output')(sfr_branch)

    # Define the model with four outputs
    model = Model(inputs=[spectral_input, continuum_input], outputs=[redshift_output, emission_output, mstar_output, sfr_output])

    return model

# Create the model
model = create_multitask_model(input_shape=(X_train.shape[1], 1), 
                               num_categories=len(bin_categories), 
                               continuum_shape=train_continuum.shape[1:])

model.summary()


# Plot the model
# plot_model(model, to_file='model_diagram.png', show_shapes=True, show_layer_names=True)


#%%
""" Class for metrics tracking """

from keras.callbacks import Callback
import matplotlib.pyplot as plt

class MetricsPlotter(Callback):
    def __init__(self, task_names, regression_tasks=None):
        self.task_names = task_names
        self.regression_tasks = regression_tasks if regression_tasks is not None else []
        self.train_loss = []
        self.val_loss = []
        self.task_metrics = {task: {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []} for task in task_names}

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        for task in self.task_names:
            self.task_metrics[task]['train_loss'].append(logs.get(f'{task}_loss'))
            self.task_metrics[task]['val_loss'].append(logs.get(f'val_{task}_loss'))
            if task != 'emission_output':  # Assuming accuracy is not relevant for emission_output
                self.task_metrics[task]['train_acc'].append(logs.get(f'{task}_accuracy'))
                self.task_metrics[task]['val_acc'].append(logs.get(f'val_{task}_accuracy'))

        self.plot_metrics(epoch)

    def plot_metrics(self, epoch):
        plt.clf()
        fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 2 rows, 3 columns

        # Total Loss
        axs[0, 0].plot(range(epoch+1), self.train_loss, label='Training Loss')
        axs[0, 0].plot(range(epoch+1), self.val_loss, label='Validation Loss')
        axs[0, 0].set_title('Total Loss')
        axs[0, 0].legend()

        # Task-specific Losses
        for i, task in enumerate(self.task_names):
            if task in self.regression_tasks:  # Plot only for regression tasks
                row, col = divmod(i, 3)
                axs[row, col].plot(range(epoch+1), self.task_metrics[task]['train_loss'], label=f'{task} Training Loss')
                axs[row, col].plot(range(epoch+1), self.task_metrics[task]['val_loss'], label=f'{task} Validation Loss')
                axs[row, col].set_title(f'{task.capitalize()} Loss')
                axs[row, col].legend()

        # Combined Accuracy Plot
        for task in self.task_names:
            if task != 'emission_output':  # Exclude emission_output
                axs[1, 2].plot(range(epoch+1), self.task_metrics[task]['train_acc'], label=f'{task} Training Accuracy', linestyle='solid')
                axs[1, 2].plot(range(epoch+1), self.task_metrics[task]['val_acc'], label=f'{task} Validation Accuracy', linestyle='dashed')
        axs[1, 2].set_title('Combined Accuracy')
        axs[1, 2].legend()

        plt.tight_layout()
        plt.show()


#%%
""" Compile & run  model """

# learning rate schedule function
def lr_schedule(epoch, lr):
    initial_lr = 0.005
    final_lr = 0.0001
    switch = 50
    if epoch < switch:
        return initial_lr - (epoch * (initial_lr - final_lr) / switch)
    else:
        return final_lr

# Add LearningRateScheduler to callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)

# compile the model
model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss={
        'redshift_output': 'categorical_crossentropy',
        'emission_output': 'binary_crossentropy',
        'mstar_output': 'mse',  # Mean Squared Error for regression
        'sfr_output': 'mse'
    },
    loss_weights={
        'redshift_output': 1,
        'emission_output': 10,
        'mstar_output': 1,
        'sfr_output': 1
    },
    metrics={
        'redshift_output': 'accuracy',
        'emission_output': 'accuracy',
        # You can add metrics for Mstar and SFR if needed
    }
)

# Early Stopping Callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To log when training is stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
)

# Instantiate the callback
metrics_plotter = MetricsPlotter(
    task_names=['redshift_output', 'emission_output', 'mstar_output', 'sfr_output'],
    regression_tasks=['mstar_output', 'sfr_output']
)



# Train the model with the callbacks
history = model.fit(
    [X_train, train_continuum],  # Training inputs
    {
        'redshift_output': Y_train_categorical,  # Training labels for redshift
        'emission_output': emission_line_masks_train,  # Training labels for emission line locations
        'mstar_output': Y_train_mstar,  # Training labels for Mstar
        'sfr_output': Y_train_sfr  # Training labels for SFR
    },
    validation_data=(
        [X_val, val_continuum],  # Validation inputs
        {
            'redshift_output': Y_val_categorical,  # Validation labels for redshift
            'emission_output': emission_line_masks_val,  # Validation labels for emission line locations
            'mstar_output': Y_val_mstar,  # Validation labels for Mstar
            'sfr_output': Y_val_sfr  # Validation labels for SFR
        }
    ),
    epochs=300,
    batch_size=1024,
    callbacks=[lr_scheduler, early_stopping, metrics_plotter]
)

#%%
""" Save Model """

model.save('current-best-model-multiTask-02Apr_2')

#%%

""" Load Model """

model = tf.keras.models.load_model('current-best-model-multiTask-02Apr_2')


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
        
        
    elif task == 'emission_output':
        # For emission mask task, plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history[task+'_loss'])
        plt.plot(history.history['val_'+task+'_loss'])
        plt.title('Model Loss for Emission Line Mask')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        if early_stopping_epoch is not None: plt.axvline(x=early_stopping_epoch, color='gray', linestyle='--')

        plt.subplot(1, 2, 2)
        plt.plot(history.history[task+'_accuracy'])
        plt.plot(history.history['val_'+task+'_accuracy'])
        plt.title('Model Accuracy for Emission Line Mask')
        plt.ylabel('Accuracy')
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
plot_history(history, 'emission_output', early_stopping_epoch=np.argmin(history.history['val_redshift_output_loss']))
plot_history(history, 'total', early_stopping_epoch=np.argmin(history.history['val_redshift_output_loss']))
plot_history(history, 'redshift_output', early_stopping_epoch=np.argmin(history.history['val_redshift_output_loss']))

#%%
""" Check global predictions on test-set """

def bin_index_to_redshift(bin_index, bin_categories):
    """ Convert a bin index back to a redshift value """
    bin_width = bin_categories[1] - bin_categories[0]
    return bin_categories[bin_index] + bin_width / 2

def inverse_transform(normalized_value, mean, std):
    return normalized_value * std + mean

def plot_predictions(predicted, actual, task_name):
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=1)
    plt.scatter(actual, predicted, alpha=0.1, s=2)
    plt.xlabel('Actual ')
    plt.ylabel('Predicted ')
    plt.title(f'Predicted vs Actual {task_name}')
    plt.show()
    

# Evaluate the model on the test set
test_metrics = model.evaluate(
    [X_test, test_continuum],  # Input features and continuum data
    {
        'redshift_output': Y_test_categorical,  # Labels for redshift task
        'emission_output': emission_line_masks_test,  # Labels for emission line mask task
        'mstar_output': Y_test_mstar,  # Labels for Mstar task
        'sfr_output': Y_test_sfr  # Labels for SFR task
    }
)

# Predictions for all tasks
all_predictions = model.predict([X_test, test_continuum])

# Unpack predictions for each task
redshift_predictions, emission_mask_predictions, mstar_predictions, sfr_predictions = all_predictions

# Get the index of the bin with the highest probability for each prediction
redshift_bin_indices = np.argmax(redshift_predictions, axis=1)

# Convert redshift bin indices back to redshift values
predicted_redshifts = np.array([bin_index_to_redshift(idx, bin_categories) + ((bin_categories[1] - bin_categories[0]) / 2) for idx in redshift_bin_indices])

# Inverse transform for Mstar and SFR predictions
mstar_predictions_rescaled = inverse_transform(mstar_predictions, mstar_mean, mstar_std)
sfr_predictions_rescaled = inverse_transform(sfr_predictions, sfr_mean, sfr_std)
Y_test_mstar_original =  inverse_transform(Y_test_mstar, mstar_mean, mstar_std)
Y_test_sfr_original = inverse_transform(Y_test_sfr, sfr_mean, sfr_std)

# Plotting
#  plot_predictions for each task
plot_predictions(Y_test, predicted_redshifts, 'Redshift')
plot_predictions(Y_test_mstar_original, mstar_predictions_rescaled, 'Mstar')  
plot_predictions(Y_test_sfr_original, sfr_predictions_rescaled, 'SFR')  



#%%
""" Check residual on specific tasks """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2 * standard_deviation ** 2))


# -------------------------------
# Calculate residuals for Mstar
# ------------------------------_
mstar_residuals = mstar_predictions_rescaled.flatten() - Y_test_mstar_original

# Create histogram for Mstar residuals
mstar_bin_heights, mstar_bin_borders, _ = plt.hist(mstar_residuals, bins=500, label='Histogram', density=True, alpha=0.6, color='b')
mstar_bin_centers = mstar_bin_borders[:-1] + np.diff(mstar_bin_borders) / 2

# Fit a Gaussian to the Mstar histogram data
mstar_popt, _ = curve_fit(gaussian, mstar_bin_centers, mstar_bin_heights, p0=[0., max(mstar_bin_heights), np.std(mstar_residuals)])

# Plot the fitted Gaussian for Mstar
mstar_x_interval_for_fit = np.linspace(mstar_bin_borders[0], mstar_bin_borders[-1], 10000)
plt.plot(mstar_x_interval_for_fit, gaussian(mstar_x_interval_for_fit, *mstar_popt), label='Fitted Gaussian', color='blue')
plt.title(f'Mstar Residuals - Gaussian: Mean = {mstar_popt[0]:.5f}, Sigma = {mstar_popt[2]:.5f}')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.show()

# -------------------------------
# Calculate residuals for SFR
# ------------------------------
sfr_residuals = sfr_predictions_rescaled.flatten() - Y_test_sfr_original

# Create histogram for SFR residuals
sfr_bin_heights, sfr_bin_borders, _ = plt.hist(sfr_residuals, bins=500, label='Histogram', density=True, alpha=0.6, color='r')
sfr_bin_centers = sfr_bin_borders[:-1] + np.diff(sfr_bin_borders) / 2

# Fit a Gaussian to the SFR histogram data
sfr_popt, _ = curve_fit(gaussian, sfr_bin_centers, sfr_bin_heights, p0=[0., max(sfr_bin_heights), np.std(sfr_residuals)])

# Plot the fitted Gaussian for SFR
sfr_x_interval_for_fit = np.linspace(sfr_bin_borders[0], sfr_bin_borders[-1], 10000)
plt.plot(sfr_x_interval_for_fit, gaussian(sfr_x_interval_for_fit, *sfr_popt), label='Fitted Gaussian', color='red')
plt.title(f'SFR Residuals - Gaussian: Mean = {sfr_popt[0]:.5f}, Sigma = {sfr_popt[2]:.5f}')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()
plt.show()

# -------------------------------
# Calculate residuals for z
# ------------------------------
# Calculate residuals
residuals = predicted_redshifts - Y_test

# Create histogram (binned data)
bin_heights, bin_borders, _ = plt.hist(residuals, bins=5000, label='Histogram', density=True, alpha=0.6, color='g')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

# Fit the Gaussian to the histogram data
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[0., max(bin_heights), np.std(residuals)])

# Plot the fitted Gaussian
x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='Fitted Gaussian', color='black')


# Annotate the plot with the Gaussian parameters
plt.title(f'Fitted Gaussian: Mean = {popt[0]:.5f}, Sigma = {popt[2]:.5f}')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()

plt.xlim(-0.2, 0.2)

plt.show()


#%%

""" residual advanced with proportions """


def plot_residuals_with_thresholds(residuals, thresholds):
    # Create histogram of residuals
    plt.figure(figsize=(10, 6))
    bin_heights, bin_borders, _ = plt.hist(residuals, bins=3000, alpha=0.6, color='g')
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    total_objects = len(residuals)  # Total number of objects in the test set

    # For each threshold, draw a vertical line and annotate the fraction
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))  # Generate distinct colors
    for threshold, color in zip(thresholds, colors):
        plt.axvline(x=threshold, color=color, linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
        
        num_objects_below_threshold = np.sum(np.abs(residuals) < threshold)
        fraction_below_threshold = num_objects_below_threshold / total_objects
        
        # Place text on the right side of the plot
        plt.text(0.9, 0.9 - 0.05 * thresholds.index(threshold), 
                 f'Residuals < {threshold} --- fraction: {fraction_below_threshold:.2f}', 
                 verticalalignment='top', horizontalalignment='right', color=color, fontsize=12, transform=plt.gca().transAxes)

    plt.xlabel('Residuals [best redshift (PDF peak) - actual redshift]')
    plt.ylabel('Number of objects - test set')
    plt.semilogx()
    plt.semilogy()
    plt.show()

# visualise
thresholds = [0.0025, 0.005, 0.01, 0.1]  # User-defined thresholds
plot_residuals_with_thresholds(residuals, thresholds)


plot_residuals_with_thresholds(residuals[df_test['MAG_vista-H'] < 25], thresholds)
plot_residuals_with_thresholds(residuals[df_test['LINE_FLUX_halpha'] > -17], thresholds)



#
#%%

""""""""""""""""""""""""""""""""
"""   post-process analyses  """
""""""""""""""""""""""""""""""""


""" check success rate vs task """

import matplotlib.pyplot as plt
import numpy as np

def analyze_success_rate_by_property_with_counts(df, X, continuum, property_name, n_bins, success_threshold, bin_categories):
    if property_name not in df.columns:
        raise ValueError(f"Property '{property_name}' not found in the dataframe.")

    # Split the property into bins
    property_values = df[property_name]
    bins = np.linspace(property_values.min(), property_values.max(), n_bins + 1)
    bin_indices = np.digitize(property_values, bins) - 1
    bin_means = (bins[:-1] + bins[1:]) / 2

    # Initialize lists for success rates and counts
    success_rates = []
    counts = []

    for i in range(n_bins):
        in_bin = bin_indices == i
        X_bin = X[in_bin]
        continuum_bin = continuum[in_bin]  # Corresponding continuum data
        actual_redshifts_bin = df['z'][in_bin]
        counts.append(np.sum(in_bin))

        if len(X_bin) == 0:
            success_rates.append(np.nan)
            continue

        redshift_distributions = model.predict([X_bin, continuum_bin])[0]
        predicted_redshift_bins = np.argmax(redshift_distributions, axis=1)
        predicted_redshifts = [bin_categories[idx] + (bin_categories[1] - bin_categories[0]) / 2 for idx in predicted_redshift_bins]

        successful_predictions = np.abs(predicted_redshifts - actual_redshifts_bin) <= success_threshold
        success_rate = np.sum(successful_predictions) / len(X_bin)
        success_rates.append(success_rate)
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)

    # Top panel for counts
    ax1.bar(bin_means, counts, width=(bins[1]-bins[0]) * 0.8, alpha=0.6, color='blue')
    ax1.set_ylabel('Number of Objects')
    ax1.set_title(f'Number of Objects in Each Bin of {property_name}')
    # ax1.set_xlim(18.5, 27.5)
    
    # Bottom panel for success rates
    ax2.step(bin_means, success_rates, where='mid', lw=4, label=f'Success Threshold = {success_threshold}')
    
    Filippo_predictions_x = [
    20.10000038, 20.29999924, 20.5, 20.70000076, 20.89999962, 21.10000038, 21.29999924, 21.5, 21.70000076, 21.89999962,
    22.10000038, 22.29999924, 22.5, 22.70000076, 22.89999962,23.10000038, 23.29999924, 23.5, 23.70000076, 23.89999962,
    24.10000038, 24.29999924, 24.5, 24.70000076, 24.89999962, 25.10000038, 25.29999924, 25.5, 25.70000076, 25.89999962]

    Filippo_predictions_y = [
    0.94897959, 0.93055556, 0.91666667, 0.91037736, 0.95084746, 0.92716763, 0.9223301, 0.92994859, 0.93858751, 0.9488013,
    0.93871688, 0.93766234, 0.9372237, 0.93686821, 0.94028882, 0.93562418, 0.92886812, 0.93595066, 0.92314252, 0.91976637,
    0.90693377, 0.88406614, 0.84163749, 0.79345111, 0.73641062, 0.66217317, 0.58544091, 0.50896019, 0.4307972, 0.37753223]
    
    ax2.plot(Filippo_predictions_x, Filippo_predictions_y, lw=4, label='Simulations by Filippo')

    ax2.set_xlabel(property_name)
    ax2.set_ylabel('Fraction of Successful Predictions')
    ax2.set_title(f'Success Rate Analysis by {property_name}')
    ax2.set_ylim(0, 1.1)
    # ax2.set_xlim(18.5, 27.5)

    ax2.legend()

    plt.tight_layout()
    plt.show()

# Example usage
# analyze_success_rate_by_property_with_counts(df_test, X_test, test_continuum, 'MAG_vista-H', n_bins=8, success_threshold=0.01, bin_categories=bin_categories)


mask = df_test['LINE_FLUX_halpha'] > -17
analyze_success_rate_by_property_with_counts(df_test[mask], X_test[mask],test_continuum[mask], 'MAG_vista-H', n_bins=8, success_threshold=0.01, bin_categories=bin_categories)



#
#%%

""""""""""""""""""""""""""""""""
"""   post-process analyses  """
""""""""""""""""""""""""""""""""


#%%
""" check individual spectra """

def plot_spectrum_with_halpha(index, X_test,test_continuum, Y_test, model, bin_categories, wavelength_axis):
    # Predict the redshift and emission lines for the selected object
    object_spectrum = X_test[index]
    object_continuum = test_continuum[index]
        
    predictions = model.predict([np.expand_dims(object_spectrum, axis=0), np.expand_dims(object_continuum, axis=0)])

    # Extract the redshift prediction - find the bin with the highest probability
    predicted_redshift_bin = np.argmax(predictions[0][0])
    predicted_redshift = bin_categories[predicted_redshift_bin] + ((bin_categories[1]-bin_categories[0])/2)

    # Actual redshift
    actual_redshift = Y_test[index] 

    # H-alpha line wavelength in Ångström (rest frame) in vacuum
    h_alpha_rest = 6564.6

    # Calculate the observed positions of the H-alpha line
    predicted_h_alpha_observed = h_alpha_rest * (1 + predicted_redshift)
    actual_h_alpha_observed = h_alpha_rest * (1 + actual_redshift)


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot the spectrum
    ax1.plot(wavelength_axis, object_spectrum/object_spectrum.max(), label='Spectrum', lw=0.5, alpha=1)

    # Plot the predicted and actual H-alpha lines
    ax1.axvline(predicted_h_alpha_observed, color='green', linestyle='-', label='Predicted H-alpha', lw=3, alpha=0.5)
    ax1.axvline(actual_h_alpha_observed, color='red', linestyle='--', label='Actual H-alpha')

    # Plot the predicted and actual emission lines
    ax1.plot(wavelength_axis, predictions[1][0]/predictions[1][0].max(), label='Predicted Emission Lines', lw=3, alpha=0.5, color='black')
    ax1.plot(wavelength_axis, emission_line_masks_test[index]/emission_line_masks_test[index].max(), '--', label='Real Emission Lines', lw=1, alpha=1, color='orange')

    ax1.set_ylabel('Intensity')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(6000, 20000)    
    ax1.set_title(f'Spectrum with Predicted and Actual H-alpha Line (Object {index})')

    # Add an inset with redshift information
    textstr = f'Predicted Redshift: {predicted_redshift:.4f}\nActual Redshift: {actual_redshift:.4f}'
    ax1.text(0.05, 0.9, textstr, transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Second subplot for redshift probability distribution
    # Adjusting the x coordinates for the bars
    bar_width = bin_categories[1] - bin_categories[0]
    adjusted_bin_categories = bin_categories + bar_width / 2
    
    # Plotting the bar chart with adjusted x coordinates
    ax3.bar(adjusted_bin_categories, predictions[0][0], width=bar_width, label='Predicted Redshift PDF')
    ax3.axvline(x=actual_redshift, color='r', linestyle='--', label='Actual Redshift')
    ax3.set_xlabel('Redshift')
    ax3.set_ylabel('Probability')
    ax3.legend()
    ax3.set_title('Predicted Redshift Probability Distribution')


    # Convert each input separately
    input_spectrum_tensor = tf.convert_to_tensor(np.expand_dims(object_spectrum, axis=0), dtype=tf.float32)
    input_continuum_tensor = tf.convert_to_tensor(np.expand_dims(object_continuum, axis=0), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Watch both tensors
        tape.watch(input_spectrum_tensor)
        tape.watch(input_continuum_tensor)
        
        # Get the prediction from the model
        prediction = model([input_spectrum_tensor, input_continuum_tensor])[0]
    
    # Compute gradients with respect to the input spectrum
    gradient = tape.gradient(prediction, input_spectrum_tensor)[0]
    processed_grad = tf.abs(gradient)
    processed_grad /= tf.math.reduce_max(processed_grad)
    processed_grad = processed_grad.numpy().flatten()  # Flatten to ensure it matches the wavelength_axis dimension

    # Ensure the dimensions match for plotting
    if processed_grad.shape[0] != wavelength_axis.shape[0]:
        raise ValueError("Saliency map and wavelength axis have different dimensions")
    
    # Plot the saliency map
    ax2.plot(wavelength_axis, processed_grad, label='Saliency Map', color='red', alpha=0.3, lw=3)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Saliency')
    ax2.set_title('Saliency Map for Redshift Prediction')
    ax2.set_xlim(6000, 20000)    
    ax2.legend()

    plt.tight_layout()
    plt.show()

# visualize
index = 2
# special_ID= 333011988000028
# index = np.where(df_test['ID'] == special_ID)[0][0]
plot_spectrum_with_halpha(index, X_test,test_continuum, Y_test, model, bin_categories, wavelength_axis)

#%%


# """ mc dropout """

# index = 100

# from tensorflow.keras import backend as K

# def mc_dropout_predict(model_path, X, n_iterations=10):
#     predictions = []
#     for _ in range(n_iterations):
#         # Clear the session to reset model state
#         # K.clear_session()
#         # Reload the model to ensure a fresh state
#         # model = tf.keras.models.load_model(model_path)
#         # Predict with dropout enabled
#         y_pred = model(X, training=True)
#         predictions.append(y_pred[0])  
#     return np.array(predictions)



# # Model path
# model_path = 'current-best-model-withNoise-line-position-classification'

# # Select a single test sample
# single_test_sample = np.expand_dims(X_test[index], axis=0)  # index is the index of your chosen test sample


# # Get MC Dropout predictions
# mc_predictions = mc_dropout_predict(model_path, single_test_sample, n_iterations=2)

# print(mc_predictions[:, 0, 0].std())

# # Visualize the results for each iteration
# plt.figure(figsize=(12, 8))
# for i in range(mc_predictions.shape[0]):
#     plt.subplot(mc_predictions.shape[0] // 2, 2, i + 1)
#     plt.bar(bin_categories, mc_predictions[i][0], width=bin_categories[1] - bin_categories[0], color='skyblue', alpha=0.7)
#     plt.xlabel('Redshift')
#     plt.ylabel('Probability')
#     plt.title(f'Iteration {i + 1}')
#     plt.xlim(0.8 ,1)
# plt.tight_layout()
# plt.show()