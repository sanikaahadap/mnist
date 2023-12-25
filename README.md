# Handwritten Digit Recognition with Convolutional Neural Networks
This repository contains code for training a Convolutional Neural Network (CNN) model using the MNIST dataset to recognize handwritten digits (0-9).

Overview
----------------
Data: The MNIST dataset, consisting of 60,000 training images and 10,000 test images of handwritten digits.
------
Model Architecture: The CNN architecture comprises three convolutional layers with ReLU activation followed by max-pooling layers, flattening to a dense layer with ReLU activation, and a final output layer using softmax activation for digit classification.
------
Training: The model is trained using TensorFlow and Keras, using the Adam optimizer and categorical cross-entropy loss over 5 epochs with a batch size of 64. A validation split of 0.2 is used during training.
------
Evaluation: After training, the model's performance is evaluated on a separate test set to determine accuracy.
