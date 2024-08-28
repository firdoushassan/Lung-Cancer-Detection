import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

import cv2
import gc
import os

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')



main_folder = r'C:\Users\iamfi\OneDrive\Desktop\Lung_Cancer_Detection_project\lung_colon_image_set\lung_image_sets'

# Creating a DataFrame with file names and labels
labels = []
file_names = []

for label in os.listdir(main_folder):
    label_folder = os.path.join(main_folder, label)
    if os.path.isdir(label_folder):
        for filename in os.listdir(label_folder):
            file_names.append(os.path.join(label, filename))
            labels.append(label)


data = pd.DataFrame({"FILE_NAME": file_names, "CATEGORY": labels})


# Spliting the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True)

# Creating a data generator for training data
train_generator = datagen.flow_from_dataframe(dataframe=train_data,
                                              directory=main_folder,
                                              x_col="FILE_NAME",
                                              y_col="CATEGORY",
                                              class_mode="categorical",
                                              target_size=(128, 128),
                                              batch_size=32)

# Displaying three sample images from each class
fig, axs = plt.subplots(len(train_generator.class_indices), 3, figsize=(15, 5 * len(train_generator.class_indices)))

for i, class_label in enumerate(train_generator.class_indices.keys()):
    class_images = train_data[train_data['CATEGORY'] == class_label]['FILE_NAME'].tolist()[:3]
    
    for j, img_path in enumerate(class_images):
        img = Image.open(os.path.join(main_folder, img_path))
        axs[i, j].imshow(img)
        axs[i, j].set_title(f"Class: {class_label}")
        axs[i, j].axis('off')

plt.show()

# Creating a data generator for testing data
test_generator = datagen.flow_from_dataframe(dataframe=test_data,
                                             directory=main_folder,
                                             x_col="FILE_NAME",
                                             y_col="CATEGORY",
                                             class_mode="categorical",
                                             target_size=(128, 128),
                                             batch_size=32)

# Building the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))  # 3 output classes

model.summary()

keras.utils.plot_model(
	model,
	show_shapes = True,
	show_dtype = True,
	show_layer_activations = True
)

# Compiling the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Defining callbacks
checkpoint = ModelCheckpoint("lung_cancer_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Training the model
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=test_generator,
                    callbacks=[checkpoint])

#Plotting Graphs
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()


# Evaluating the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Generating predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Printing classification report and confusion matrix
print("Classification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

print('Model Saved!')

