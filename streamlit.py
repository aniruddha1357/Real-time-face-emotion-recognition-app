
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Streamlit app title
st.title("Human Face Detection")

# Upload an image
uploaded_image = st.file_uploader("Upload Your Image", type=["jpg", "png", "jpeg"])

# Checkboxes for analysis tasks
age_checkbox = st.checkbox("Age")
gender_checkbox = st.checkbox("Gender")
emotion_checkbox = st.checkbox("Emotion")

# Display the uploaded image and perform analysis
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform analysis when "Analyze" button is clicked
    if st.button("Analyze"):
        tasks = []
        if age_checkbox:
            tasks.append('age')
        if gender_checkbox:
            tasks.append('gender')
        if emotion_checkbox:
            tasks.append('emotion')

        # Resize the image to 224x224 pixels and convert to RGB
        img_pil_resized = image.resize((224, 224), Image.LANCZOS)
        img_pil_resized = img_pil_resized.convert("RGB")

        img_cv = np.array(img_pil_resized)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        result = DeepFace.analyze(img_cv, actions=tasks)

        # Display the analysis results
        for task in tasks:
            if task == 'emotion':
                st.write(f"{task.capitalize()}: {result[0]['dominant_emotion']}")
            elif task == 'age':
                st.write(f"{task.capitalize()}: {result[0]['age']}")
            elif task == 'gender':
                st.write(f"{task.capitalize()}: {result[0]['gender']}")