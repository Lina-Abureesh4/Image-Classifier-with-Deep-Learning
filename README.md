# 🌸 Image Classifier with Deep Learning

This project is part of the **Intro to Machine Learning with TensorFlow Nanodegree**. It demonstrates how to build and train a deep neural network for image classification using TensorFlow and Keras. The model is trained on a flower dataset and later converted into a command-line application that can predict the type of flower from a given image.

## 🚀 Project Overview

The project consists of two main parts:

### Part 1: Jupyter Notebook Model Development
- File: `Project_Image_Classifier_Project.ipynb`
- Trained a deep learning model using a pre-trained network (transfer learning).
- Enabled GPU acceleration locally to speed up training.
- Processed and loaded the flower dataset.
- Fine-tuned the model and evaluated its accuracy.

### Part 2: Command Line Application
- Files:
  - `predict.py`: A script that loads a trained model and predicts the class of a given flower image.
  - `utility.py`: Contains helper functions for image preprocessing and model loading.
  - `projUdacity.h5`: The trained Keras model saved from Part 1.
- Users can provide an image and receive the top-k predicted flower classes with probabilities.

## 🛠️ Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- argparse (for command-line arguments)

## 🧠 What I Learned

- How to apply transfer learning using pre-trained models.
- How to handle and preprocess image data.
- Saving and loading Keras models.
- Converting a trained model into a user-friendly command-line application.
- Structuring Python projects and utility modules.

## 📄 Acknowledgements

This project was completed as part of the [Intro to Machine Learning with TensorFlow Nanodegree](https://www.udacity.com/course/intro-to-machine-learning-with-tensorflow--ud187),  
under the **Palestine Launch Pad Scholarship** sponsored by **Google** and **Udacity**.

