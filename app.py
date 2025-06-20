import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tempfile
import os

# Load model
model = load_model("emotion_model.keras")
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Preprocess face
def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)  # add channel for grayscale
    return face, (x, y, w, h)

# Predict emotion
def predict_emotion(image):
    result = preprocess_face(image)
    if result is None:
        return "No face", None
    face_input, (x, y, w, h) = result
    prediction = model.predict(face_input, verbose=0)
    confidence = np.max(prediction)
    label = labels[np.argmax(prediction)]
    return f"{label} ({confidence:.2f})", (x, y, w, h)

# Streamlit UI
st.title("🎥 Real-Time Emotion Detection")
option = st.radio("Select Input Type:", ["Upload Image", "Use Webcam", "Upload Video"])

# Image upload
if option == "Upload Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if img_file:
        image = Image.open(img_file)
        img_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        emotion, _ = predict_emotion(img_np)
        st.success(f"Prediction: {emotion}")

# Webcam mode
elif option == "Use Webcam":
    st.info("Use the button below to capture an image from your webcam.")
    
    # Take photo using browser camera
    captured_image = st.camera_input("Take a picture")

    if captured_image is not None:
        # Load and convert image to NumPy array
        image = Image.open(captured_image).convert('RGB')
        img_np = np.array(image)

        # Convert RGB to BGR for OpenCV (since you're using cv2 face detector)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Predict emotion
        emotion, box = predict_emotion(img_bgr)

        # Draw bounding box and emotion
        if box:
            x, y, w, h = box
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img_np, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the result image with overlay
        st.image(img_np, caption=f"Detected Emotion: {emotion}", use_container_width=True)
    else:
        st.warning("Please take a photo to begin prediction.")


# Video upload
elif option == "Upload Video":
    vid_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if vid_file:
        temp_path = tempfile.NamedTemporaryFile(delete=False).name + ".mp4"
        with open(temp_path, "wb") as f:
            f.write(vid_file.read())

        st.video(vid_file)

        st.info("Processing video for emotion predictions...")
        cap = cv2.VideoCapture(temp_path)
        frame_display = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            emotion, box = predict_emotion(frame)
            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        os.remove(temp_path)

