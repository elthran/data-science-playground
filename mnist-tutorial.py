import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras

num_classes = 10
input_shape = (28, 28, 1)


def display_images_and_labels(images, labels, is_prediction=False):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        # Extract each image from the data
        image = images[i]
        image = image / 255

        # Extract the given or predicted class of the image
        image_class = np.argmax(labels[i]) if is_prediction else labels[i]

        plt.imshow(image, cmap='gray', interpolation='nearest')
        if is_prediction:
            plt.title("class={}".format(image_class))
        else:
            plt.title("class={}".format(image_class))
        plt.axis('off')
    plt.show()


# Each image is a 28x28 matrix, where each value is between 0-255
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

display_images_and_labels(images=x_train, labels=y_train)
x_test_raw = x_test

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1) by creating a new axis
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

assert x_train.shape == (60000, 28, 28, 1)
assert x_test.shape == (10000, 28, 28, 1)

# Convert class vectors to binary class matrices.
# Ie. instead of each target being a number, it's an array of our 10 possible predictions
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [keras.Input(shape=input_shape), keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
     keras.layers.MaxPooling2D(pool_size=(2, 2)), keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
     keras.layers.MaxPooling2D(pool_size=(2, 2)), keras.layers.Flatten(), keras.layers.Dropout(0.5),
     keras.layers.Dense(num_classes, activation="softmax"), ])

print("Here is the summary of our new model")
model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]}\n Test accuracy: {score[1]}")

display_images_and_labels(images=x_test_raw, labels=y_test, is_prediction=True)
