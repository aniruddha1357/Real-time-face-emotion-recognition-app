import streamlit as st
import cv2
from PIL import Image
from deepface import DeepFace
import numpy as np

# Streamlit app title
st.title("Real-time Face Analysis")

# Checkbox for analysis tasks
analysis_options = st.checkbox("Show Analysis Options", True)
age_checkbox = st.checkbox("Age")
gender_checkbox = st.checkbox("Gender")
emotion_checkbox = st.checkbox("Emotion")

# Initialize camera
cap = cv2.VideoCapture(0)

# Run the streamlit app
while st.button("Analyze"):
    # Read a frame from the camera
    ret, frame = cap.read()

    if frame is not None:
        # Display the original frame
        st.image(frame, channels="BGR", use_column_width=True)

        tasks = []
        if analysis_options:
            if age_checkbox:
                tasks.append('age')
            if gender_checkbox:
                tasks.append('gender')
            if emotion_checkbox:
                tasks.append('emotion')

        # Perform face analysis
        result = DeepFace.analyze(frame, actions=tasks)

        # Display the analysis results
        for task in tasks:
            if task == 'emotion':
                st.write(f"{task.capitalize()}: {result[0]['dominant_emotion']}")
            elif task == 'age':
                st.write(f"{task.capitalize()}: {result[0]['age']}")
            elif task == 'gender':
                st.write(f"{task.capitalize()}: {result[0]['gender']}")
