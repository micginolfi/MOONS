#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02 Apr 24

@author: mginolfi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, concatenate, BatchNormalization, Activation, Multiply, Permute, BatchNormalization, Attention
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
""" utils to make emissionLines-masks """

# Example emission lines' rest wavelengths in AA
# I took Vacuum values (https://classic.sdss.org/dr7/products/spectra/vacwavelength.php)
# conversion here: https://www.sdss3.org/dr8/spectro/spectra.php
emission_lines_rest = {
    'OII_1': 3727.1,            # Oxygen II
    'OII_2':  3729.9,            # Oxygen II (https://www.aanda.org/articles/aa/pdf/2013/11/aa22452-13.pdf)
    'H_beta': 4862.7,        # Hydrogen-beta
    'OIII_1': 4960.3,           # Oxygen III
    'OIII_2': 5008.2,           # Oxygen III
    'NII_1': 6549.8,            # Nitrogen II
    'H_alpha': 6564.6,       # Hydrogen-alpha
    'NII_2': 6585.3,            # Nitrogen II
    'SII_1': 6718.3,            # Sulfur II
    'SII_2': 6732.7,            # Sulfur II
    'SIII_1': 9070,            # Sulfur III
    'SIII_2': 9532,            # Sulfur III
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
z = 3 # Example redshift
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

df_train = pd.read_pickle('train_dataset.pickle')

# mask_restrict = df_train['MAG_vista-H'] < 24
# df_train = df_train[mask_restrict]

# Extract features for training set
import pickle
with open('noise_all_spectra_train_normalized.pkl', 'rb') as file:
    all_spectra_train_normalized = pickle.load(file)

all_spectra_train_normalized[:, mask_sky] = 0
# all_spectra_train_normalized = all_spectra_train_normalized[mask_restrict]

X_train = all_spectra_train_normalized

# Extract labels for training set
all_redshift_train = df_train['z'].values

# define labels: make Y
Y_train = all_redshift_train

emission_line_masks_train = np.array([create_emission_output(z, wavelength_axis, emission_lines_rest) for z in all_redshift_train])

# Convert labels to categorical
Y_train_categorical = create_categorical_labels(Y_train, sigma=0)

del df_train

#%%
""" load validation set, make X & Y dataset and normalise"""

df_val = pd.read_pickle('validation_dataset.pickle')

# mask_restrict = df_val['MAG_vista-H'] < 24
# df_val = df_val[mask_restrict]

# Extract features for validation set
import pickle
with open('noise_all_spectra_val_normalized.pkl', 'rb') as file:
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

del df_val


#%%

#mic qui unisco val e train (a questo punto le val loss non sono piu significative)

X_train = np.concatenate((X_train, X_val))
Y_train_categorical = np.concatenate((Y_train_categorical, Y_val_categorical))
emission_line_masks_train = np.concatenate((emission_line_masks_train, emission_line_masks_val))


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

#%%
""""""""""""""""""""""""""""""""
"""   Modelling              """
""""""""""""""""""""""""""""""""

from keras.layers import Input, Conv1D, Dense, Flatten, Dropout, Concatenate
from keras.models import Model

def create_multitask_model(input_shape, num_categories):
    # Shared layers
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=16, kernel_size=11, activation='selu')(inputs)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(filters=32, kernel_size=11, activation='selu')(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(filters=64, kernel_size=11, activation='selu')(x)
    x = MaxPooling1D(pool_size=5)(x)

    shared_features = Flatten()(x)

    # Emission line location prediction branch
    emission_branch = Dense(128, activation='selu')(shared_features)
    emission_branch = Dropout(0.2)(emission_branch)
    emission_branch = Dense(256, activation='selu')(emission_branch)
    emission_branch = Dropout(0.2)(emission_branch)

    emission_output = Dense(input_shape[0], activation='sigmoid', name='emission_output')(emission_branch)


    # Combine shared features with emission line information for redshift prediction
    combined_features = Concatenate()([shared_features, emission_branch])

    # Redshift prediction branch
    redshift_branch = Dense(128, activation='selu')(combined_features)
    redshift_branch = Dropout(0.2)(redshift_branch)
    redshift_branch = Dense(256, activation='selu')(redshift_branch)
    redshift_branch = Dropout(0.2)(redshift_branch)

    redshift_output = Dense(num_categories, activation='softmax', name='redshift_output')(redshift_branch)

    # Define the model with two outputs
    model = Model(inputs=inputs, outputs=[redshift_output, emission_output])
    return model

# Create the model
model = create_multitask_model(input_shape=(X_train.shape[1], 1), num_categories=len(bin_categories))

model.summary()


#%%
""" Class for metrics tracking """

from keras.callbacks import Callback


class MetricsPlotter(Callback):
    def __init__(self, task_names):
        self.task_names = task_names  # List of task names
        self.train_loss = []
        self.val_loss = []
        self.task_metrics = {task: {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []} for task in task_names}

    def on_epoch_end(self, epoch, logs=None):
        # Update total and task-specific metrics
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        for task in self.task_names:
            self.task_metrics[task]['train_loss'].append(logs.get(f'{task}_loss'))
            self.task_metrics[task]['val_loss'].append(logs.get(f'val_{task}_loss'))
            self.task_metrics[task]['train_acc'].append(logs.get(f'{task}_accuracy'))
            self.task_metrics[task]['val_acc'].append(logs.get(f'val_{task}_accuracy'))

        self.plot_metrics(epoch)

    def plot_metrics(self, epoch):
        plt.clf()
        rows = 2
        cols = len(self.task_names)
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

        # Total Loss
        axs[0, 0].plot(range(epoch+1), self.train_loss, label='Training Loss')
        axs[0, 0].plot(range(epoch+1), self.val_loss, label='Validation Loss')
        axs[0, 0].set_title('Total Loss')
        axs[0, 0].legend()

        for i, task in enumerate(self.task_names):
            # Task-specific Loss
            axs[1, i].plot(range(epoch+1), self.task_metrics[task]['train_loss'], label=f'{task} Training Loss')
            axs[1, i].plot(range(epoch+1), self.task_metrics[task]['val_loss'], label=f'{task} Validation Loss')
            axs[1, i].set_title(f'{task.capitalize()} Loss')
            axs[1, i].legend()

            # Task-specific Accuracy
            if i < cols - 1:  # Check to avoid IndexError
                axs[0, i+1].plot(range(epoch+1), self.task_metrics[task]['train_acc'], label=f'{task} Training Accuracy')
                axs[0, i+1].plot(range(epoch+1), self.task_metrics[task]['val_acc'], label=f'{task} Validation Accuracy')
                axs[0, i+1].set_title(f'{task.capitalize()} Accuracy')
                axs[0, i+1].legend()

        plt.tight_layout()
        plt.show()

#%%
""" Compile & run  model """

# learning rate schedule function
def lr_schedule(epoch, lr):
    initial_lr = 0.0005
    final_lr = 0.00001
    switch = 50
    if epoch < switch:
        return initial_lr - (epoch * (initial_lr - final_lr) / switch)
    else:
        return final_lr

# Add LearningRateScheduler to callbacks
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss={'redshift_output': 'categorical_crossentropy', 'emission_output': 'binary_crossentropy'},
              loss_weights={'redshift_output': 1, 'emission_output': 1.5},
              metrics={'redshift_output': 'accuracy', 'emission_output': 'accuracy'})


# Early Stopping Callback
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To log when training is stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
)

#  callback
metrics_plotter = MetricsPlotter(task_names=['redshift_output', 'emission_output'])

# Train the model with the callbacks
history = model.fit(
    X_train,
    {'redshift_output': Y_train_categorical, 'emission_output': emission_line_masks_train},
    validation_data=(X_val, {'redshift_output': Y_val_categorical, 'emission_output': emission_line_masks_val}),
    epochs=300,
    batch_size=1024,
    callbacks=[lr_scheduler, early_stopping, metrics_plotter]
)

#%%
""" Save Model """

model.save('current-best-model-withNoise-line-position-classification-Martedi20Feb')

#%%

""" Load Model """

# model = tf.keras.models.load_model('current-best-model-noNoise-line-position')


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
    plt.xlabel('Actual Redshift')
    plt.ylabel('Predicted Redshift')
    plt.title(f'Predicted vs Actual {task_name}')
    plt.show()

# Evaluate the model on the test set
test_metrics = model.evaluate(X_test, {'redshift_output': Y_test_categorical, 'emission_output': emission_line_masks_test})

# Predictions
predictions = model.predict(X_test)
redshift_predictions = predictions[0]
emission_mask_predictions = predictions[1]

# Get the index of the bin with the highest probability for each prediction
redshift_bin_indices = np.argmax(redshift_predictions, axis=1)

# Convert these indices back to redshift values
predicted_redshifts = np.array([bin_index_to_redshift(idx, bin_categories) for idx in redshift_bin_indices])

# Plot redshift predictions
plot_predictions(predicted_redshifts, Y_test, 'Redshift')

#%%
""" Check residual on specific tasks """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2 * standard_deviation ** 2))

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

#
#%%

""""""""""""""""""""""""""""""""
"""   post-process analyses  """
""""""""""""""""""""""""""""""""


#%%
""" check individual spectra """

def plot_spectrum_with_halpha(index, X_test, Y_test, model, bin_categories, wavelength_axis):
    # Predict the redshift and emission lines for the selected object
    object_spectrum = X_test[index]
    predictions = model.predict(np.expand_dims(object_spectrum, axis=0))

    # Extract the redshift prediction - find the bin with the highest probability
    predicted_redshift_bin = np.argmax(predictions[0])
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


    # Compute the saliency map for the redshift prediction
    input_sample_tensor = tf.convert_to_tensor(np.expand_dims(object_spectrum, axis=0), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_sample_tensor)
        prediction = model(input_sample_tensor)[0]
    gradient = tape.gradient(prediction, input_sample_tensor)[0]
    if np.any(np.isnan(gradient.numpy())):
        print(f"Gradient contains NaN values for input at index {index}.")
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
plot_spectrum_with_halpha(index, X_test, Y_test, model, bin_categories, wavelength_axis)

#%%




""" mc dropout """

index = 100

from tensorflow.keras import backend as K

def mc_dropout_predict(model_path, X, n_iterations=10):
    predictions = []
    for _ in range(n_iterations):
        # Clear the session to reset model state
        # K.clear_session()
        # Reload the model to ensure a fresh state
        # model = tf.keras.models.load_model(model_path)
        # Predict with dropout enabled
        y_pred = model(X, training=True)
        predictions.append(y_pred[0])
    return np.array(predictions)



# Model path
model_path = 'current-best-model-withNoise-line-position-classification'

# Select a single test sample
single_test_sample = np.expand_dims(X_test[index], axis=0)  # index is the index of your chosen test sample


# Get MC Dropout predictions
mc_predictions = mc_dropout_predict(model_path, single_test_sample, n_iterations=2)

print(mc_predictions[:, 0, 0].std())

# Visualize the results for each iteration
plt.figure(figsize=(12, 8))
for i in range(mc_predictions.shape[0]):
    plt.subplot(mc_predictions.shape[0] // 2, 2, i + 1)
    plt.bar(bin_categories, mc_predictions[i][0], width=bin_categories[1] - bin_categories[0], color='skyblue', alpha=0.7)
    plt.xlabel('Redshift')
    plt.ylabel('Probability')
    plt.title(f'Iteration {i + 1}')
    plt.xlim(0.8 ,1)
plt.tight_layout()
plt.show()
