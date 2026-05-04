import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


FILTERS = ["None", "Blur", "Edges", "Threshold", "Grayscale"]


def decode_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    return image


def apply_filter(image_bgr, filter_name):
    if filter_name == "Blur":
        return cv2.GaussianBlur(image_bgr, (15, 15), 0)

    if filter_name == "Edges":
        edges = cv2.Canny(image_bgr, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if filter_name == "Threshold":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if filter_name == "Grayscale":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return image_bgr


def segment_person(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_selfie = mp.solutions.selfie_segmentation

    with mp_selfie.SelfieSegmentation(model_selection=1) as segment:
        results = segment.process(image_rgb)

    if results.segmentation_mask is None:
        return None

    mask = results.segmentation_mask > 0.5
    person_only = np.zeros_like(image_bgr)
    person_only[mask] = image_bgr[mask]
    return person_only


st.set_page_config(page_title="AI Face Segmentation & Filters", layout="centered")
st.title("AI Face Segmentation & Filters")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
filter_option = st.selectbox("Choose Filter", FILTERS)

if uploaded_file is not None:
    try:
        original_bgr = decode_uploaded_image(uploaded_file)
        segmented_bgr = segment_person(original_bgr)

        if segmented_bgr is None:
            st.error("MediaPipe could not create a segmentation mask for this image.")
        else:
            filtered_bgr = apply_filter(segmented_bgr, filter_option)

            st.subheader("Original")
            st.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), channels="RGB")

            st.subheader("Segmentation + Filter")
            st.image(cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB), channels="RGB")

    except AttributeError as exc:
        if "solutions" in str(exc):
            st.error(
                "MediaPipe did not load correctly. Check that no local file is named "
                "mediapipe.py and reinstall mediapipe in the active virtual environment."
            )
        else:
            st.error(f"Unexpected MediaPipe error: {exc}")
    except Exception as exc:
        st.error(f"Could not process the uploaded image: {exc}")
