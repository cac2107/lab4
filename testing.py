import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from tensorflow.keras import layers, models

# Step 1: Preprocess the data
def preprocess_data(data_folder):
    images = []
    labels = []
    
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(folder_path, filename), 'r') as file:
                        lines = file.readlines()
                        # Example parsing of the text file
                        gender = lines[0].split(': ')[1].strip()
                        label = lines[1].split(': ')[1].strip()
                        # You can extract other relevant information from the text file if needed
                        labels.append(label)
                    
                elif filename.endswith(".png"):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (512, 512))  # Resize if necessary
                    images.append(img)
    
    return np.array(images), np.array(labels)

data_folder = 'C:\\Users\\Cole\\Documents\\SEM 6\\Auth\\lab4\\data\\sd04\\png_txt\\'  # Change this to the parent folder containing subfolders
images, labels = preprocess_data(data_folder)

# Step 2: Define a model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # num_classes depends on your classification labels
])

# Step 3: Train the model
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
