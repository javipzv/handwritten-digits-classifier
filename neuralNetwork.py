from silence_tensorflow import silence_tensorflow
silence_tensorflow()

####################################

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Necesitamos que sean de una dimensión
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000,28*28))

# Reescalamos los datos en escala 0 - 1
train_images = train_images.astype("float32")/255
test_images = test_images.astype("float32")/255

# Codificar las labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Creamos la red
oculta1 = tf.keras.layers.Dense(units=512, activation="relu", input_shape = (28*28, ))
salida = tf.keras.layers.Dense(units=10, activation="softmax")
model = tf.keras.Sequential([oculta1, salida])

# Compilamos el modelo
model.compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
    )

# Training
hist = model.fit(train_images,
         train_labels,
         epochs = 25,
         batch_size = 128)

model.save("model")

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_loss: ", test_loss, "\ntest_acc: ", test_acc)

plt.xlabel("Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(hist.history["loss"])
plt.show()