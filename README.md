# Handwritten Digits Classifier

## Description
This code defines a GUI (graphical user interface) for a digit classifier. It allows the user to draw a digit on a canvas and classify 
the digit using a pre-trained TensorFlow model. The model is based on a simple neural net with only 1 hidden layer of 512 neurons.


## How to use
To use it, you only have to run the 'digitClassifier.py'. When run, the GUI will display a canvas that allows you to draw a digit with the mouse.
You can then save the drawn image by clicking the "Save" button, and the image will be preprocessed and fed into the model to classify the digit.
The predicted digit will be displayed in a label. The "Delete" button can be used to clear the canvas and start over.

## About the model
The model in the repository is a model based on a neural net. It is a simple feedforward neural network with two dense layers. 
The first layer has 512 units with a ReLU activation function. This layer has an input shape of (28*28,), which suggests that it is designed to process input images that are 28x28 pixels in size, flattened into a 1D array of length 784.
The second layer has 10 units with a softmax activation function. This layer is designed to produce a probability distribution over the 10 possible classes (0-9) for the input image.
The neural network is compiled with the RMSprop optimizer and categorical cross-entropy loss function. 
The accuracy metric is also specified for evaluating the performance of the model during training.
Finally, the model is trained on a dataset of images and labels using the fit() method for 25 epochs and a batch size of 128.
All this information can be seen in the 'neuralNetwork.py' file, and, of course, it can be changed to try new parameters combinations.

## Visual
  
<img width="1050" alt="GUI" src="https://user-images.githubusercontent.com/90279135/234730629-dd2d4a0a-b997-4dca-9902-795010e34103.png">
