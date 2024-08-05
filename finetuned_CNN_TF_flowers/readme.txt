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

### Setting Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Here's how you can set it up:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

To run the project, simply execute the Jupyter notebook provided in the repository. The notebook includes steps for data preprocessing, model fine-tuning, and evaluation.

```bash
jupyter notebook finetuned_CNN_TF_flowers_98.31%_acc.ipynb
```

### Data Loading and Preprocessing

The notebook begins by loading and preprocessing the flower dataset. Ensure your dataset is organized and accessible for loading.

### Example Code for Data Loading

```python
import tensorflow as tf

data_dir = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

def build_dataset(subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        subset=subset,
        label_mode="categorical",
        seed=123,
        image_size=(512, 512),
        batch_size=32)
```

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

model = tf.keras.Sequential([
    hub.KerasLayer(model_handle, input_shape=(512, 512, 3), trainable=True),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(build_dataset('training'),
                    epochs=5,
                    validation_data=build_dataset('validation'))
```

## Model Architecture

The architecture of the fine-tuned CNN model is based on EfficientNetV2. The model leverages transfer learning to adapt the pre-trained weights to the flower classification task. Key components include:

- EfficientNetV2 backbone for feature extraction
- Custom dense layers for classification

### Example of Model Architecture Code

```python
model = tf.keras.Sequential([
    hub.KerasLayer(model_handle, input_shape=(512, 512, 3), trainable=True),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
```

## Results

After fine-tuning the model, the accuracy achieved on the validation set is an impressive 98.31%. Detailed evaluation metrics and visualizations, such as confusion matrices, are included in the notebook.

### Example Code for Evaluating the Model

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
