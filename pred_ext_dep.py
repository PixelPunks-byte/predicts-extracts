import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained MobileNetV2 model with the classification head
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load the custom pixelpunk02.h5 model
pixelpunk_model = tf.keras.models.load_model("pixelpunk02.h5")

# Mapping of weapon indices to names
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

# Function to classify and print the label using both models
def classify_and_print_label(image_path):
    image = preprocess_image(image_path)
    if image is not None:
        # Predict using MobileNetV2 model
        mobilenet_predictions = mobilenet_model.predict(image)
        decoded_mobilenet_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(mobilenet_predictions, top=4)[0]
        mobilenet_labels = []
        for i, (class_id, class_name, score) in enumerate(decoded_mobilenet_predictions):
            if score > 0.01:
                mobilenet_labels.append(f"{i + 1}: {class_name} ({score:.2f})")
        
        # Predict using pixelpunk02 model
        pixelpunk_predictions = pixelpunk_model.predict(image)
        predicted_index = np.argmax(pixelpunk_predictions, axis=-1)[0]  # Get the index with the highest score
        predicted_weapon = weapon_names.get(predicted_index, "Unknown Weapon")
        
        return mobilenet_labels, predicted_weapon

# Streamlit app
st.title("Image Classification and Feature Detection")
st.write("Upload an image to classify and detect its features:")

st.markdown(
    f"""
    <style>
    {open("style.css").read()}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    mobilenet_labels, predicted_weapon = classify_and_print_label(image_path)
    
    # Display the results
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        st.write("Detected Features:")
        for label in mobilenet_labels:
            st.write(label)
    with col2:
        st.write("Image Prediction:")
        st.write(f"{predicted_weapon}")

    # Add a horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)