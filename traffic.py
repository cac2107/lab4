import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 4
IMG_WIDTH = 512
IMG_HEIGHT = 512
NUM_CATEGORIES = 5
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    subDirs = []
    images = []
    labels = []

    cat_dict = {'A': 0,
                'L': 1,
                'R': 2,
                'T': 3,
                'W': 4}

    for path in next(os.walk(data_dir))[1]:
        subDirs.append(path)

    subDirs.sort()

    i = 0
    for path in subDirs:
        if i == 3: break
        p = os.path.join(data_dir, str(path))
        for image in os.listdir(p):
            if image[-4:] == ".txt": continue
            try:
                with open(f"{p}\\{image[:-4]}.txt", 'r') as f:
                    text = f.read().split("\n")[1].split(" ")[1]
            except Exception as e:
                print(f"Path: {p} {image}\nError:{e}")
                continue

            labels.append(cat_dict[text])
            im = cv2.imread(os.path.join(data_dir, str(path), image), 3)
            resizedImage = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT))
            images.append(resizedImage)
        i += 1
    finalTup = (images, labels)
    return finalTup


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(120, 3, padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(70, 3, padding='same', activation='relu'),
        tf.keras.layers.Dropout(.15),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        # tf.keras.layers.Conv2D(80, 3, padding='same', activation='relu'),
        # tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(640, activation='relu'),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CATEGORIES)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    main()
