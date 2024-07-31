# Fine-Tuned CNN for Flower Classification

Welcome to the Flower Classification project! This repository contains a deep learning model fine-tuned using TensorFlow and TensorFlow Hub for classifying flower images. The project achieves an impressive accuracy of 98.31%.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project fine-tunes a pre-trained EfficientNetV2 model using a dataset of flower images. The model is designed to accurately classify various species of flowers, leveraging the powerful transfer learning capabilities of TensorFlow Hub.

## Installation

To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/flower-classification.git
cd flower-classification
pip install -r requirements.txt
```

## Usage

To run the project, simply execute the Jupyter notebook provided in the repository. The notebook includes steps for data preprocessing, model fine-tuning, and evaluation.

```bash
jupyter notebook finetuned_CNN_TF_flowers_98.31%_acc.ipynb
```

### Data Loading and Preprocessing

The notebook begins by loading and preprocessing the flower dataset. Ensure your dataset is organized and accessible for loading.

### Model Fine-Tuning

The model is fine-tuned using TensorFlow Hub's pre-trained EfficientNetV2 model. The following code snippet shows how to load the pre-trained model and fine-tune it:

```python
import tensorflow as tf
import tensorflow_hub as hub

model_name = "efficientnetv2-xl-21k-ft1k"
model_handle_map = {"efficientnetv2-xl-21k-ft1k": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2"}
model_image_size_map = {"efficientnetv2-xl-21k": 512}
model_handle = model_handle_map[model_name]

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
```

## Model Architecture

The architecture of the fine-tuned CNN model is based on EfficientNetV2. The model leverages transfer learning to adapt the pre-trained weights to the flower classification task. Key components include:

- EfficientNetV2 backbone for feature extraction
- Custom dense layers for classification

## Results

After fine-tuning the model, the accuracy achieved on the validation set is an impressive 98.31%. Detailed evaluation metrics and visualizations, such as confusion matrices, are included in the notebook.

```python
# Example code for evaluating the model
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()
