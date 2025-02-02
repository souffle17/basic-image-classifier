import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk
import keras._tf_keras.keras

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def select_model():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            global model
            model = keras.saving.load_model(file_path)
            result_label.config(text="Ready")
        except:
            result_label.config(text="Failed to load model")
        
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return class_names[predicted_class[0]]


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((150, 150))  # just for display, not for processing
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        try:
            prediction = predict_image(file_path)
            result_label.config(text=f"Guess: {prediction}")
        except:
            result_label.config(text="Error")
    
root = tk.Tk()
root.title("Image Classifier")

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack()

model_button = tk.Button(root, text="Select .keras File", command=select_model)
model_button.pack()

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

root.mainloop()