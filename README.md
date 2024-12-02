
# Image Data Analysis
## Project Description

This project focuses on the [FashionMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset, which serves as a more exciting and challenging variant of the well-known [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset of handwritten digits.

The dataset consists of grayscale images with dimensions of 28x28 pixels, containing 60,000 training images and 10,000 test images.


### Project Objectives

 1. **Neural Network Construction**

The first part of the project requires building a neural network using the [Keras library](https://keras.io/), achieving an accuracy of at least 85% on the [FashionMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  test set. The neural network will be trained on a sample of 10,000 training images for efficiency.

  

2. **Image Processing and Object Recognition**

The second part involves additional image processing using the [OpenCV library](https://opencv.org/) to recognize multiple clothing items in a single image. Several test cases have been prepared for this purpose, consisting of 512x512 images on a white background (with noise) featuring approximately 10 clothing items. The items may be randomly scaled but are always correctly oriented.

The [neural network](https://en.wikipedia.org/wiki/Neural_network_%28machine_learning%29) identifies all clothing items in each test case, drawing bounding boxes around them in blue. Additionally, the program displays the class of each clothing item in red.

  

## Requirements

  

- [Python 3.x](https://www.python.org/downloads/)

- [Keras](https://keras.io/)

- [TensorFlow](https://www.tensorflow.org/)

- [OpenCV](https://opencv.org/)

- [NumPy](https://numpy.org/)

- [Matplotlib](https://matplotlib.org/) (optional, for visualizations)

  

## Installation

  

1. Clone this repository:

```bash

git clone https://github.com/lmicovic/neural-network-image-recognition.git

```

  

2. Navigate to the project directory:

```bash

cd project

```

  

3. Install the required packages:

```bash

pip install -r requirements.txt

```

4. Verify that dependencies are installed:

```bash

pip list

```
