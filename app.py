import streamlit as st
from model import build_unet
import numpy as np
from tensorflow.keras.layers import Input
import cv2
from io import BytesIO

SIZE_X = 256  # Set your desired size
SIZE_Y = 256  # Set your desired size
 
import streamlit as st
st.set_page_config(
    page_title="Cancer Detection",
    page_icon="🔬",
    layout="centered",)


st.title("Cancer Detection with AI")
st.write("Upload an MRI image to detect brain tumors.")

# File Upload Section
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

# Display Uploaded Image and Processing Section
if uploaded_file is not None:
    st.sidebar.title("Uploaded Image")
    
    # Save the uploaded image to a temporary file
    temp_file_path = 'temp_uploaded_image.png'
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    # Read the image using cv2
    uploaded_image = cv2.imread(temp_file_path, cv2.IMREAD_GRAYSCALE)

    st.sidebar.image(uploaded_image, caption='Input (Gray)', width=200, channels="GRAY")

    # Resize the image to (SIZE_Y, SIZE_X)
    img = cv2.resize(uploaded_image, (SIZE_Y, SIZE_X))

    # Convert the grayscale image to a 3-channel image with identical channels
    img = cv2.merge([img, img, img])

    # Build and Load Model
    input_layer = Input((SIZE_X, SIZE_Y, 3))
    model = build_unet(input_layer, 'he_normal', 0.2)
    model.load_weights('trained_model\checkpointFUZZY_RES_3.hdf5')

    # Add an extra dimension to represent the batch size
    img_with_batch = np.expand_dims(img, axis=0)

    # Model Prediction
    res = model.predict(img_with_batch)

    # Invert the intermediate mask
    inverted_mask = 1 - res[0, :, :, 0]

    # Choose a threshold for binary segmentation
    threshold = 0.98  # Adjust this value based on your model's output characteristics
    segmentation_mask = (inverted_mask > threshold).astype(np.uint8)
    has_tumor = np.any(segmentation_mask)

    # Display the final model output segmentation mask
    st.image(segmentation_mask * 255, caption='Output Segmentation Mask', width=200, channels="GRAY")

    # Result Section
    st.sidebar.title("Model Output")

    if has_tumor:
        st.sidebar.success("✅ Tumor detected!")
        st.success("✅ Tumor detected! Please consult with a medical professional.")
    else:
        st.sidebar.info("❌ No tumor detected.")
        st.info("❌ No tumor detected. The brain appears healthy.")

# Adding Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #ffffff;'>© 2024 GDSC_Cancer-detection. All Rights Reserved.</p>", unsafe_allow_html=True)
