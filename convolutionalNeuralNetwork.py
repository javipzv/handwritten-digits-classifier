import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape the data
train_images = train_images.astype("float32")/255
test_images = test_images.astype("float32")/255

# Labels to categorical values
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Create the network
CNN = Sequential()
CNN.add(Conv2D(32, (4, 4), input_shape=(28, 28, 1), activation='relu', padding='same'))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Dropout(0.1))
CNN.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Dropout(0.1))
CNN.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
CNN.add(MaxPooling2D(pool_size = (2, 2)))
CNN.add(Dropout(0.1))
CNN.add(Flatten())
CNN.add(Dense(128, activation='relu'))
CNN.add(Dense(10, activation='softmax'))
CNN.summary()

# Compile the model
CNN.compile(optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"])

# Training
hist = CNN.fit(train_images,
                train_labels,
                epochs = 10,
                batch_size = 128)

# CNN.save("model_CNN")

# Evaluating the model
test_loss, test_acc = CNN.evaluate(test_images, test_labels)
print("test_loss: ", test_loss, "\ntest_acc: ", test_acc)

# Viewing results
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(hist.history["loss"])
plt.show()

# Effect of the first convolution filters
from scipy import signal
test_img = test_images[1].reshape((28, 28))

# Filters
plt.figure(figsize = (18, 8))
for i in range(32):
  filter = CNN.layers[0].get_weights()[0][:, :, :, i].reshape((4, 4))
  plt.subplot(4, 8, i+1)
  plt.imshow(filter, cmap='gray')
plt.show()

# Convolutions
plt.figure(figsize = (18, 8))
for i in range(32):
  filter = CNN.layers[0].get_weights()[0][:, :, :, i].reshape((4, 4))
  res = signal.correlate2d(test_img, filter)
  plt.subplot(4, 8, i+1)
  plt.imshow(res, cmap='gray')
plt.show()