import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image


def predict_single_image(model, img_path):
    # Loading and preprocessing the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizing the image

    # Making predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    
    # Decoding the predicted class index to the original label
    class_labels = {0: 'adenocarcinoma', 1: 'normal', 2: 'squamous'}
    predicted_class = class_labels[predicted_class_index]
    
    # Display the image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class_index, predicted_class

if __name__ == "__main__":
    # Path to the trained model file
    model_path = r"C:\Users\iamfi\OneDrive\Desktop\Lung_Cancer_Detection_project\VGG16\lung_cancer_model_vgg16.h5"  

    # Loading the trained model
    model = keras.models.load_model(model_path)

    # Getting the path to the single image
    #image_path = r"C:\Users\iamfi\OneDrive\Desktop\Lung_Cancer_Detection_project\lung_colon_image_set\lung_image_sets\lung_adenocarcinoma\lungaca2578.jpeg"

# =============================================================================
    #image_path = r"C:\Users\iamfi\OneDrive\Desktop\Lung_Cancer_Detection_project\lung_colon_image_set\lung_image_sets\lung_normal\lungn789.jpeg"
# =============================================================================
    
# =============================================================================
    image_path = r"C:\Users\iamfi\OneDrive\Desktop\Lung_Cancer_Detection_project\lung_colon_image_set\lung_image_sets\lung_squamous\lungscc3739.jpeg"
# =============================================================================
        
    # Making prediction for the single image
    predicted_index, predicted_class = predict_single_image(model, image_path)

    print(f"Predicted Class Index: {predicted_index}")
    print(f"Predicted Class Name: {predicted_class}")
