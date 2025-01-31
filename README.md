### Project: Image Classification (Cat vs Dog)

## Project Description

This is a final project for the machine learning module, aimed at creating and training a model to classify images into two categories: **cat** or **dog**. The model is implemented in Python using **TensorFlow** (Keras) for training deep neural networks. Additionally, we use **scikit-learn** (for data splitting and class weight calculation), matplotlib and seaborn (for visualization), as well as unittest (for unit testing).

We use a classic convolutional neural network (CNN) model with multiple **Conv2D** layers, **MaxPooling2D** layers, a **Flatten** layer, and dense (**Dense**) layers. The final layer is a single-unit output layer with a **sigmoid** activation function to distinguish between the two classes (cat/dog).

## Code Structure

The project is divided into several classes, each responsible for different elements of processing, training, and prediction:

**1. Config**

Stores configuration settings: data paths, training parameters (number of epochs, batch size, learning rate, etc.), as well as paths for saving the trained model and TensorBoard logs.

**2. DataProcessor**

Loads data from folders ("Cat" and "Dog") and splits the dataset into training and testing sets.

Normalizes images so that pixel values are within the range .

Includes functions for data analysis (e.g., displaying basic statistics or pixel value histograms).

**3. FeatureEngineer**

Responsible for feature engineering. In this case, it prepares a data generator (ImageDataGenerator) and performs image augmentation (shifts, rotations, flips, zoom, etc.), which helps artificially expand the training dataset and improve overall model accuracy.

**4. ModelTrainer**

Builds the neural network architecture (Sequential model with convolutional layers, pooling layers, and fully connected layers).

Compiles the model with the Adam optimizer and binary_crossentropy loss function (suitable for binary classification problems).

Trains the model on training data, including validation, model checkpoint saving, and logging in TensorBoard.

Provides a function to evaluate the model's performance on the test set.

Also offers a mechanism for fine-tuning the model.

**5. App**

The main class that integrates all the above components into a complete application pipeline:

Loads and splits data into training and test sets.

Performs basic data analysis (e.g., pixel value histogram).

Augments training data.

Builds and trains the model, followed by fine-tuning.

Evaluates the model on the test set and displays results (e.g., confusion matrix, classification report).

**6. ModelPrediction**

A helper class that loads the trained model from a .keras file.

Responsible for loading a single image, preprocessing it (resizing, normalizing), and making a prediction (returning the probability of the "Dog" or "Cat" class).

## Program Workflow

**1. Checking the Data Path**
The program first verifies whether the dataset folder (PetImages) exists and contains the Cat and Dog subfolders.

**2. Loading and Splitting Data**

All images from the Cat and Dog folders are loaded.

Images are resized to 128×128 and normalized.

The dataset is split into training and test sets (e.g., in an 80% / 20% ratio).

**3. Data Analysis**

Basic statistics are displayed (number of samples, label distribution, pixel value mean and standard deviation, histogram).

4. **Data Augmentation**

Training data is passed through an augmentation generator (shifts, rotations, zoom changes, and other transformations).

This generates additional samples, helping to prevent overfitting.

**5. Model Building and Compilation**

A Sequential model is created with multiple convolutional and pooling layers, followed by dense layers (including a Dropout layer for regularization).

The final layer is a single-unit (Dense(1)) with sigmoid activation, enabling binary classification (cat/dog).

**6. Model Training**

The model is trained on the training set (with or without augmentation, depending on settings).

During training, accuracy and loss are monitored for both training and validation sets.

The best model (with the highest validation accuracy) is saved to a .keras file.

Logs are saved in the logs folder, allowing them to be analyzed in TensorBoard.

**7. Fine-Tuning the Model**

Layers can be unfrozen and the model fine-tuned with a lower learning rate, often improving its performance.

**8. Model Evaluation**

The final accuracy and loss are measured on the test set.

A confusion matrix and classification report (precision, recall, f1-score) are displayed to assess performance for both classes (cat/dog).

**9. New Image Prediction (Optional)**

After training and saving the model, it can be loaded and tested on a new image that was not part of the dataset.

The model returns a "Dog" or "Cat" label along with the predicted probability.

## Required Libraries

tensorflow version: 2.18.0 (preferably version 2.x)

numpy version: 1.26.4

matplotlib version: 3.8.4

seaborn version: 0.13.2

scikit-learn version: 1.4.2

unittest

! [Testing Various Layer Combinations in TensorBoard](tensorboard.png)


We saved logs of several trained models in TensorBoard and, based on the Loss value, selected the best model with the following layers:
```python
Input(shape=(*self.config.image_size, 3)),  # Input layer
Conv2D(32, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Dropout(0.3),
Conv2D(128, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.3),
Dense(64, activation='relu'),
Dense(32, activation='relu'),
Dense(1, activation='sigmoid')
```
