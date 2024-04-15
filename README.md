# Handwritten Digits Classifier

## Description
This code defines a GUI (graphical user interface) for a digit classifier. It allows the user to draw a digit on a canvas and classify the digit using a pre-trained TensorFlow model. The model is based on a simple neural net with only 1 hidden layer of 512 neurons.

## How to use
To use it, you only have to run the 'digitClassifier.py'. When run, the GUI will display a canvas that allows you to draw a digit with the mouse. You can then save the drawn image by clicking the "Save" button, and the image will be preprocessed and fed into the model to classify the digit. The predicted digit will be displayed in a label. The "Delete" button can be used to clear the canvas and start over. Additionally, the "Change Model" button allows you to switch between the simple neural network model and a convolutional neural network model.

## About the models

### Simple Neural Network Model
The model in the repository is a simple feedforward neural network with two dense layers. The first layer has 512 units with a ReLU activation function. This layer has an input shape of (28*28,), which suggests that it is designed to process input images that are 28x28 pixels in size, flattened into a 1D array of length 784. The second layer has 10 units with a softmax activation function. This layer is designed to produce a probability distribution over the 10 possible classes (0-9) for the input image. The neural network is compiled with the RMSprop optimizer and categorical cross-entropy loss function. The accuracy metric is also specified for evaluating the performance of the model during training. Finally, the model is trained on a dataset of images and labels using the fit() method for 25 epochs and a batch size of 128.

### Convolutional Neural Network (CNN) Model
In addition to the simple neural network model, a convolutional neural network (CNN) model has been implemented. The CNN model comprises three convolutional layers followed by max-pooling and dropout layers to extract and learn features from input images. The model then flattens the output and passes it through two dense layers for classification. It is compiled using the Adam optimizer and categorical cross-entropy loss function and trained on a dataset of images for 10 epochs and a batch size of 128. This CNN model is designed to capture spatial dependencies in the input images more effectively. As expected, the CNN model performs slightly better than the simple neural network model. You can switch between these models using the "Change Model" button in the GUI.

## Visual
  
<img width="569" alt="Captura del modelo" src="https://github.com/javipzv/handwritten-digits-classifier/assets/90279135/8117ac9d-3ad9-4ac9-81b7-17e5ce052d8b">

## Conclusions
The performance advantage of the Convolutional Neural Network (CNN) over a traditional neural network in the digits recognition task is relatively marginal. This slight improvement can be attributed to the CNN's ability to effectively capture spatial dependencies in the input images, thanks to its convolutional and pooling layers. However, the task of recognizing handwritten digits, while a vision problem, is relatively straightforward even for a standard neural network architecture due to the simplicity of the dataset and the distinct features present in each digit. Let's visualize the results:


