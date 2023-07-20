## Introduction

One of the successful branches of artificial intelligence is `computer vision`, which allows computer to gain some insights from digital images and/or video. `Neural networks` can be successfully used for computer vision tasks.

Imagine you're developing a system to recognize printed text. You've used some algorithmic approach to align the page and cut out individual characters in the text, and now you need to recognize individual letters. This problem is called `image classification`, because we need to separate input images into different classes. Other examples of such a problem would be automatically sorting post-cards according to the image, or determining product type in a delivery system from a photograph.

In this module, we'll learn how to train image classifiction neural network models using `PyTorch`, one of the most popular Python libraries for building neural networks. We'll start from simplest model - a fully connected dense neural network - and from a simple MNIST dataset of handwritten digits. We'll then learn about `convolutional neural networks`,
which are designed to capture 2D image patterns, and switch to more complex dataset, CIFAR-10. Finally, we'll use `pre-trained networks` and `transfer learning` to allow us to train models on realatively small datasets.

By the end of this module, you'll be able to train image classification models on real-world photographs, such as cats and dogs dataset, and develop image classifiers for your own scenarios.


## Learning Objectives

* Learn about computer vision tasks most commonly solved with neural networks
* Understand how Convolutional Neural Networks (CNNs) work
* Train a neural network to recognize handwritten digits and classify cats and dogs
* Learn how to use Transfer Learning to solve real-world classification problems with PyTorch

## Notebooks
|No|Title|Kaggle|
|---|---|---|
|1|[Introduction to CV with PyTorch](computer_vision/introduction_to_cv_with_pytorch.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/introduction-to-computer-vision-with-pytorch)|
|2|[Training a simple sense neural network](computer_vision/training_a_simple_cnn.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/training-a-simple-dense-neural-network)|
|3|[Convolutional Neural Networks](computer_vision/use_a_convolutional_neural_network)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/use-a-convolutional-neural-networks)|
|4|[Multilayer Dense Neural Network](computer_vision/training_multi_layer_convolutional_neural_network.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/multilayer-dense-convolutional-neural-network/notebook)|
|5|[Pre-trained models and transfer learning](computer_vision/pre_trained_models_and_transfer_learning.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/pre-trained-models-and-transfer-learning)|
|6|[Lightweight Networks and MobileNet](computer_vision/lightweight_networks_and_mobileNet.ipynb)|[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/aisuko/lightweight-networks-and-mobilenet)|

## Prerequisites

* Basic knowledge of Python and Jupyter Notebooks
* Familiarity with PyTorch framework, including tensors, basics of back propagation and building models
* Understanding machine learning concepts, such as classification, train/test dataset, accuracy, etc.


## Takeaways

We can know how convolutional nerual networks work and how they can capture patterns in 2D images. In addition, CNNs can also be used for finding patterns in 1-dimentional signals (such as sound waves, or time series), and in multi-dimensional structures like events in video, where some patterns are repated across frames.

Also, CNNs are simple building blocks for solving more complex computer vision tasks, such as Image Generation. **Generative Adversarial Networks** can be used to generate images similar to the ones in the given dataset. For example, like computer-generated paintings. Similarly, CNNs are used for object detection, instance segmentation.
