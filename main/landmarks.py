import cv2
import numpy as np
from typing import List, Iterable
from mediapipe.python.solutions.face_mesh import FaceMesh

def detect_landmarks(src: np.ndarray, is_stream: bool = False):
    """
    Given an image `src` retrieves the facial landmarks associated with it
    """

    # detection_confidence = app.sidebar.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.5, label_visibility="hidden")

    with FaceMesh(static_image_mode=not is_stream, max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as face_mesh:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(src)
    if results.multi_face_landmarks: # type: ignore
        return results.multi_face_landmarks[0].landmark # type: ignore


def normalize_landmarks(landmarks, height: int, width: int, mask: Iterable = None): # type: ignore
    """
    The landmarks returned by mediapipe have coordinates between [0, 1].
    This function normalizes them in the range of the image dimensions so they can be played with.
    """

    normalized_landmarks = np.array(
        [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks])
    if mask:
        normalized_landmarks = normalized_landmarks[mask] # type: ignore
    return normalized_landmarks


def plot_landmarks(src: np.ndarray, landmarks: List, show: bool = False): # type: ignore
    """
    Given a source image and a list of landmarks plots them onto the image
    """
    dst = src.copy()
    for x, y in landmarks:
        cv2.circle(dst, (x, y), 2, 0, cv2.FILLED)
    if show:
        print("Displaying image plotted with landmarks")
        # cv2.imshow("Plotted Landmarks", dst)
    return dst
