import streamlit as st
import numpy as np

# Load h5 model 
import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('converted_keras/keras_model.h5')

# Load labels
labels = open('converted_keras\labels.txt')
labels = labels.read().split('\n')

# Input image
st.title("Padi Classifier")
uploaded_file = st.file_uploader("Upload Gambar Padi", type="jpg")
if uploaded_file is not None:
    image = tf.io.decode_image(uploaded_file.getvalue(), channels=3, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis=0)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Predict
    prediction = model.predict(image)
    predicted_class = labels[np.argmax(prediction)]
    # prediksi = np.argmax(prediction)
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {np.max(prediction*100, axis=1)[0]:.2f}%")



