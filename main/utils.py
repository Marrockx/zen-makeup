import cv2
import numpy as np
from landmarks import detect_landmarks, normalize_landmarks, plot_landmarks
from mediapipe.python.solutions.face_detection import FaceDetection

upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408,
            415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320,
            307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
face_conn = [10, 338, 297, 332, 284, 251, 389, 356,454, 323,361,
            288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 
            140, 150, 136, 172, 58,  132, 93, 234, 127, 162, 21, 
            54, 103, 67, 109]
cheeks = [425, 205]
left_brow = [ 70, 63, 105, 66, 107, 55, 65, 52, 53]
right_brow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 300]





