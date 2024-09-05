import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained MobileNetV2 model with the classification head
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the custom pixelpunk02.h5 model (for weapon detection)
pixelpunk_model = tf.keras.models.load_model("pixelpunk02.h5")

# Mapping of weapon indices to names (for the custom model)
weapon_names = {
    0: "Automatic Rifle",
    1: "Bazooka",
    2: "Grenade Launcher",
    3: "Handgun",
    4: "Knife",
    5: "SMG",
    6: "Shotgun",
    7: "Sniper",
    8: "Sword"
}

# List of weapon-related class names from MobileNetV2
weapon_keywords = ["rifle", "assault_rifle", "bazooka", "pistol", "knife", "gun", "shotgun", "sword", "grenade"]

# Function to preprocess the input image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure the image is in RGB format
        image = image.resize((224, 224))  # Resize to the size expected by the model
        image_np = np.array(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_np, axis=0).astype(np.float32)  # Add batch dimension and convert to float32
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return None

# Function to classify with pixelpunk model (weapon detection)
def predict_with_custom_model(image):
    custom_prediction = pixelpunk_model.predict(image)
    predicted_index = np.argmax(custom_prediction, axis=-1)[0]
    predicted_label = weapon_names.get(predicted_index, "Unknown Weapon")
    return predicted_label

# Function to classify with MobileNetV2 model (general image classification)
def predict_with_mobilenet(image):
    mobilenet_prediction = mobilenet_model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(mobilenet_prediction, top=4)[0]
    return decoded_predictions

# Combined prediction function
def combined_predict(image_path):
    image = preprocess_image(image_path)
    
    if image is None:
        return "Error in processing image."
    
    # Predict with MobileNetV2 and get top 4 features
    mobilenet_predictions = predict_with_mobilenet(image)
    
    # Check if top 2 of top 4 features are weapon-related
    weapon_related = False
    for _, class_name, _ in mobilenet_predictions:
        if any(keyword in class_name for keyword in weapon_keywords):
            weapon_related = True
            break
    
    if weapon_related:
        # If weapons are detected, use the custom weapon detection model
        predicted_weapon = predict_with_custom_model(image)
        return mobilenet_predictions, predicted_weapon
    else:
        # If not weapon-related, return MobileNetV2's general classification
        return mobilenet_predictions, "N/A"

# Create a Streamlit app

st.markdown("<h1 style='text-align: center;'>Image Classification App</h1>", unsafe_allow_html=True)
st.write("Upload an image to classify:")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_path = uploaded_file.name
    image_data = uploaded_file.getvalue()
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # Classify the image
    mobilenet_labels, predicted_weapon = combined_predict(image_path)
    
    # Display the results
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        st.write("Detected Features:")
        for i, (class_id, class_name, score) in enumerate(mobilenet_labels):
            st.write(f"{i + 1}: {class_name} ({score:.2f})")
    with col2:
        st.write("Weapon Prediction:")
        st.write(predicted_weapon)