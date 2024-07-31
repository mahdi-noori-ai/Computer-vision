# CIFAR-10 Classification Project

This project contains a deep learning model for image classification using the CIFAR-10 dataset. It utilizes TensorFlow and Keras libraries to build and train a convolutional neural network (CNN) for classifying images into 10 different categories.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. This project aims to build a CNN model to achieve high accuracy in image classification tasks.

## Files and Directories

- `Deep Learning classification - CIFAR10-0.73%.ipynb`: The main Jupyter notebook containing the code for data loading, preprocessing, model building, training, and evaluation.
- `requirements.txt`: A file listing the dependencies required to run the project.

## Installation

To set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/cifar10-classification.git
    cd cifar10-classification
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, open the Jupyter notebook and execute the cells. The notebook includes detailed steps for the entire workflow:

1. Data Loading and Preprocessing
2. Model Architecture Definition
3. Model Training
4. Model Evaluation

### Data Loading and Preprocessing

The notebook begins by downloading and splitting the CIFAR-10 dataset into training and test sets:

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
