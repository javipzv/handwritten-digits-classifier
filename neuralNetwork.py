import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Input layer of one dimension
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

# Scale the data
train_images = train_images.astype("float32")/255
test_images = test_images.astype("float32")/255

# Labels to categorical values
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Creating the neural network
oculta1 = tf.keras.layers.Dense(units=512, activation="relu", input_shape = (28*28, ))
salida = tf.keras.layers.Dense(units=10, activation="softmax")
NN = tf.keras.Sequential([oculta1, salida])

# Compile the model
NN.compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"])

# Training
hist = NN.fit(train_images,
         train_labels,
         epochs = 25,
         batch_size = 128)

# NN.save("model_NN")

# Evaluating the model
test_loss, test_acc = NN.evaluate(test_images, test_labels)
print("test_loss: ", test_loss, "\ntest_acc: ", test_acc)

# Viewing results
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(hist.history["loss"])
plt.show()