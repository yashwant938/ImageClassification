# Classification App

## Overview
This Kotlin program collects a set of images and uses a convolutional neural network (CNN) to classify them. It leverages TensorFlow Lite, a lightweight machine learning framework designed for mobile and embedded devices, to perform image classification tasks efficiently on Android.

## Features
* Collect images: Allows users to select a set of images from the device's gallery.
* Image classification: Utilizes a pre-trained convolutional neural network model to classify images into predefined categories.
* TensorFlow Lite integration: Uses TensorFlow Lite to run the neural network model efficiently on Android devices.
* Real-time classification: Performs image classification in real-time, providing instant feedback to the user.

## Usage
1. Tap the "Select Images" button to choose images from the device's gallery for classification.
2. Once images are selected, tap the "Get Predictions" button to classify each image using the pre-trained CNN model.
3. View the classification results displayed on the screen for each image.
4. To remove selected images and start over, tap the "Remove Images" button.

## Requirements
* TensorFlow Lite runtime library integrated into the application.
* Functional access to the device's gallery for selecting images.

## Installation
1. Clone or download the repository to your local machine.
2. Open the project in Android Studio.
3. Build and run the application on your Android device.

## Technology Stack
* Kotlin: Programming language used for Android application development.
* TensorFlow Lite: Lightweight machine learning framework for mobile and embedded devices.
* Android: Operating system for mobile devices.