import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageFont  # Import ImageFont module
import tkinter as tk
from tkinter import filedialog

def predict_single_image(model, img_path, label, panel):
    # Loading and preprocessing the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizing the image

    # Making predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    
    # Decoding the predicted class index to the original label
    class_labels = {0: 'Adenocarcinoma', 1: 'Normal', 2: 'Squamous'}
    predicted_class = class_labels[predicted_class_index]
    
    # Display the image
    img = Image.open(img_path)
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    img_width = int(0.5 * window_width)
    img_height = int(0.5 * window_height)
    img = img.resize((img_width, img_height), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk  # Keep a reference to the PhotoImage object

    # Update the predicted class label
    label.config(text=f"Predicted Class: {predicted_class}", font=("Helvetica", 18, "bold"), fg="black")
    label.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        predict_single_image(model, file_path, predicted_label, image_panel)

if __name__ == "__main__":
    # Path to the trained model file
    model_path = r"C:\Users\iamfi\OneDrive\Desktop\Lung_Cancer_Detection_project\VGG16\lung_cancer_model_vgg16.h5"  

    # Loading the trained model
    model = keras.models.load_model(model_path)

    # Creating a Tkinter window
    root = tk.Tk()
    root.title("Lung Cancer Prediction")

    # Background image
    bg_image = Image.open("image.jpeg")
    window_width = root.winfo_screenwidth()
    window_height = root.winfo_screenheight()
    bg_image = bg_image.resize((window_width, window_height), Image.ANTIALIAS)
    draw = ImageDraw.Draw(bg_image)
    font = ImageFont.truetype("freedom.ttf", 52)  # Load font
    draw.text((window_width/2, window_height/7), "LUNG CANCER DETECTION MODEL", fill=(255, 255, 255, 128), anchor="ma", font=font)

    # Convert the background image to PhotoImage
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Image panel
    image_panel = tk.Label(root, bg="#f0f0f0")
    image_panel.pack(expand=True)

    # Button to browse image
    browse_button = tk.Button(root, text="Browse Image", command=browse_image, font=("Helvetica", 20), bg="#172547", fg="#ffffff")
    browse_button.pack(pady=(40, 20))

    # Predicted class label
    predicted_label = tk.Label(root, text="", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
    predicted_label.pack(pady=(2,2))

    root.geometry("1000x600")
    root.mainloop()
