import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# ---------------- UI ----------------
st.set_page_config(page_title="Face App", layout="centered")
st.title("Face Detection + Shape Recognition")

# ---------------- SETUP ----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

face_cap = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FILTER ----------------
def apply_filter(frame, mode):
    if mode == "bright":
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    elif mode == "gray":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    elif mode == "sharp":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

    return frame

# ---------------- FACE SHAPE ----------------
def estimate_shape(landmarks, w, h):
    left = landmarks[234].x * w
    right = landmarks[454].x * w
    top = landmarks[10].y * h
    chin = landmarks[152].y * h

    if right - left == 0:
        return "Unknown"

    ratio = (chin - top) / (right - left)

    if ratio > 1.6:
        return "Oval"
    elif ratio > 1.45:
        return "Heart"
    elif ratio > 1.3:
        return "Diamond"
    elif ratio > 1.1:
        return "Round"
    else:
        return "Square"

# ---------------- UI CONTROLS ----------------
filter_type = st.selectbox("Choose Filter", ["normal", "bright", "gray", "sharp"])

# Camera input (WEB SAFE)
img_file_buffer = st.camera_input("Take a photo")

# ---------------- PROCESS ----------------
if img_file_buffer is not None:

    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    if filter_type != "normal":
        frame = apply_filter(frame, filter_type)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray, 1.1, 5)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    shape = "Detecting..."

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            shape = estimate_shape(
                face_landmarks.landmark,
                frame.shape[1],
                frame.shape[0]
            )

    # draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, f"Shape: {shape}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(frame, channels="BGR")

else:
    st.info("Press the camera button and take a photo")