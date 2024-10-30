# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import math
import numpy as np
from featureExtractors import FEATURE_NAMES
import matplotlib.pyplot as plt

PRINT = True


class PerceptronPacman:

    def __init__(self, num_train_iterations=20, learning_rate=1):

        self.max_iterations = num_train_iterations
        self.learning_rate = learning_rate

        # A list of which features to include by name. To exclude a feature comment out the line with that feature name
        feature_names_to_use = [
            'closestFood', 
            # 'closestFoodNow',
            'closestGhost',
            'closestGhostNow',
            'closestScaredGhost',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            # 'eatsFood',
            "foodCount",
            # 'foodWithinFiveSpaces',
            # 'foodWithinNineSpaces',
            # 'foodWithinThreeSpaces',  
            # 'furthestFood', 
            'numberAvailableActions',
            # "ratioCapsuleDistance",
            # "ratioFoodDistance",
            "ratioGhostDistance",
            "ratioScaredGhostDistance"
            ]
        
        # we start our indexing from 1 because the bias term is at index 0 in the data set
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(1, len(FEATURE_NAMES) + 1)))
        # a list of the indices for the features that should be used. We always include 0 for the bias term.
        self.features_to_use = [0] + [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]
        print(f'Length of feature names to use: {len(feature_names_to_use)}, Features to use: {self.features_to_use}')

        "*** YOUR CODE HERE ***"
        self.training_accuracies = []  # To store training accuracies
        self.validation_accuracies = []  # To store validation accuracies


        # SLP
        self.input_size = len(self.features_to_use)  # Number of input features
        self.weights = np.random.randn(self.input_size, 1) # Weights for each feature, shape (input_size, 1)
        self.bias = np.zeros((1, 1))  # Bias for the output, shape (1, 1)

    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your perceptron and output activation function

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.
        """
        # filter the data to only include your chosen features. We might not need to do this if we're working with training data that has already been filtered.
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        "*** YOUR CODE HERE ***"
        # SLP
        if feature_vector.ndim > 1:
            if feature_vector.shape[1] > len(self.features_to_use):
                vector_to_classify = feature_vector[:, self.features_to_use]
            else:
                vector_to_classify = feature_vector
        else:
            vector_to_classify = feature_vector[self.features_to_use]
        # vector_to_classify = feature_vector[:, self.features_to_use] if feature_vector.ndim > 1 else feature_vector[self.features_to_use]
        # vector_to_classify = feature_vector
        # print(f'Feature vector: {feature_vector}, {type(feature_vector)}, Vector to classify: {vector_to_classify.shape}')
        # Calculate the raw output (for batches, use matrix multiplication)
        raw_output = np.dot(vector_to_classify, self.weights) + self.bias

        # Apply activation function (sigmoid) and return the predictions
        return self.activationOutput(raw_output)


    def activationHidden(self, x):
        """
        Implement your chosen activation function for any hidden layers here.
        """

        "*** YOUR CODE HERE ***"
        # ReLU function
        return np.maximum(0, x)

    def activationOutput(self, x):
        """
        Implement your chosen activation function for the output here.
        """

        "*** YOUR CODE HERE ***"
        # Sigmoid function to represent how strong the state is (0 to 1). (SLP)
        return 1 / (1 + np.exp(-x))

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here. 
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        # print(f'Data shape: {data.shape}, Length of data: {len(data)}, Features to use: {self.features_to_use}')
        # X_eval = data[:, self.features_to_use]
        # if len(data) > len(self.features_to_use):
        #     X_eval = data[:, self.features_to_use]
        # else:
        #     X_eval = data

        "*** YOUR CODE HERE ***"
        
        if data.shape[1] > len(self.features_to_use):
            X_eval = data[:, self.features_to_use]
        else:
            X_eval = data

        # print(f'X_eval shape: {X_eval.shape}')
        predictions = self.predict(X_eval) > 0.5  # Classify based on a 0.5 threshold
        labels = labels.to_numpy().reshape(-1, 1)
        # print(f'Data: {data}, Data shape: {data.shape}, Data type: {type(data)}')
        # print(f'Labels: {labels}, Type: {type(labels)}')
        # print(f'Predictions shape: {predictions.shape}, Labels shape: {labels.shape}')
        accuracy = np.mean(predictions == labels)
        return accuracy

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        X_train = trainingData[:, self.features_to_use]
        X_validate = validationData[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        # SLP
        # Convert labels to numpy arrays if needed and reshape to match output shape
        y_train = np.array(trainingLabels).reshape(-1, 1)
        
        for iteration in range(self.max_iterations):
            # Forward pass: calculate the perceptron output
            linear_output = np.dot(X_train, self.weights) + self.bias
            output = self.activationOutput(linear_output)

            # Compute error between predicted output and true labels
            output_error = output - y_train

            # Backward pass: update weights and bias using gradient descent
            self.weights -= self.learning_rate * np.dot(X_train.T, output_error) / len(X_train)
            self.bias -= self.learning_rate * np.mean(output_error)

            # Optional: Calculate and print loss (binary cross-entropy) for monitoring
            if PRINT and iteration % 10 == 0:
                loss = -np.mean(y_train * np.log(output + 1e-15) + (1 - y_train) * np.log(1 - output + 1e-15))
                # print(f'X_train shape: {X_train.shape}')
                train_accuracy = self.evaluate(X_train, trainingLabels)  # Training accuracy
                val_accuracy = self.evaluate(X_validate, validationLabels) # Validation accuracy
                self.training_accuracies.append(train_accuracy)
                self.validation_accuracies.append(val_accuracy)
                # print(f'Iteration {iteration}, Loss: {loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

        self.plot_accuracies()

    def save_weights(self, weights_path):
        """
        Saves your weights to a .model file. You're free to format this however you like.
        For example with a single layer perceptron you could just save a single line with all the weights.
        """
        "*** YOUR CODE HERE ***"
        # SLP
        with open(weights_path, 'w') as file:
            # Flatten weights and bias and write them to the file
            np.savetxt(file, self.weights.flatten(), delimiter=",")
            np.savetxt(file, self.bias.flatten(), delimiter=",")
        print(f"Weights and bias saved to {weights_path}")

        
    def load_weights(self, weights_path):
        """
        Loads your weights from a .model file. 
        Whatever you do here should work with the formatting of your save_weights function.
        """
        "*** YOUR CODE HERE ***"
        # SLP
        with open(weights_path, 'r') as file:
            # Load weights and bias from the file
            weights_flat = np.loadtxt(file, delimiter=",", max_rows=self.weights.size)
            bias_flat = np.loadtxt(file, delimiter=",", max_rows=1)

        # Reshape weights and bias to their original shapes
        self.weights = weights_flat.reshape(self.weights.shape)
        self.bias = bias_flat.reshape(self.bias.shape)
        print(f"Weights and bias loaded from {weights_path}")

    def plot_accuracies(self):
        # Accuracies only saved every 10 iterations
        x_values = list(range(0, self.max_iterations, 10))
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, self.training_accuracies, label='Training Accuracy', color='b')
        plt.plot(x_values, self.validation_accuracies, label='Validation Accuracy', color='r')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracies Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()