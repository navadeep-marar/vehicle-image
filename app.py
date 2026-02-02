import streamlit as st
import numpy as np
from PIL import Image
import pickle
import cv2

# Title
st.title("Vehicle Classification App 🚗🛵✈️")

# Load the trained model from .pkl
with open("vehicle_model.pkl", "rb") as file:
    model = pickle.load(file)

# List of labels (must match the model training order)
labels = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img_size = 224
    img = image.convert("RGB")
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    img_array = img_array / 255.0  # rescale
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Display result
    st.success(f"Predicted Class: {labels[class_index]}")
    st.info(f"Confidence: {confidence*100:.2f}%")
