# AR MAKEUP VISUALIZATION APPLICATION INTERFACE WITH STREAMLIT

#   comment to remove in lowercase

import streamlit as app
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image, ImageColor
import csv

from landmarks import *
from apply_makeup import *

from utils import *
from regions import makeup_segments

from color_conversion import *

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh



# app.title('ZEN UP🎨')
app.title('').markdown("<div style='background-color:#160926; margin-top:-40px; margin-left:-80px; margin-right:-80px; height:80px; display: flex; justify-content: center; align-items: center;'><h3 style='font-family: Inter; color:#ffffff; font-size: 36px; font-weight: 700; text-transform: uppercase'>ZEN UP 🎨</h3></div>",
                    unsafe_allow_html=True)  # 💄

app.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #DBC7F4;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

app.sidebar.title('INTERACTIONS')
app.sidebar.markdown('---')

app.sidebar.subheader('MAKEUP REGIONS')

@app.cache_resource()

def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        # r = ()
        dim = (int((w * width) / float(w)), height)
    else:
        r = width/float(w)
        dim = (width, int(h * r));

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

makeup_region = app.sidebar.selectbox('Select Region to Apply Makeup', makeup_segments)

app.sidebar.markdown('---')
app.sidebar.subheader('SHADES')

app.sidebar.write(f"<h5 style='text-align: center; color: black; text-align: left;'>Select Colour to Apply</h5>", unsafe_allow_html=True)

makeup = [];

if makeup_region is not None:
    if (makeup_region == 'LEFT EYEBROW ONLY'):
        left_eyebrow_colour = app.sidebar.color_picker(
        "LEFT EYEBROW FILL COLOUR", value="#ffff00", key="LEY")
        makeup.append({"name": "lbrow", "color": rgb_to_bgr(hex_to_rgb(left_eyebrow_colour))})

    if (makeup_region == 'RIGHT EYEBROW ONLY'):
        right_eyebrow_colour = app.sidebar.color_picker(
        "RIGHT EYEBROW FILL COLOUR", value="#ff00ff", key="REY")
        makeup.append({"name": "rbrow", "color": rgb_to_bgr(hex_to_rgb(right_eyebrow_colour))})

    if ('EYEBROWS' in makeup_region):
        eyebrows_colour = right_eyebrow_colour = app.sidebar.color_picker(
        "EYEBROWS FILL COLOUR", value="#fffff0", key="EY")
        makeup.append({"name": "lbrow", "color": rgb_to_bgr(hex_to_rgb(eyebrows_colour))})
        makeup.append({"name": "rbrow", "color": rgb_to_bgr(hex_to_rgb(eyebrows_colour))})

    if ('CHEEKS' in makeup_region):
        blush_colour = app.sidebar.color_picker(
        "BLUSH COLOUR", value="#fff010", key="CH")
        makeup.append({"name": "blush", "color": rgb_to_bgr(hex_to_rgb(blush_colour))})

    if ('FACE BOUNDARY' in makeup_region):
        foundation_colour = app.sidebar.color_picker(
        "FOUNDATION COLOUR", value="#00ff10", key="FB")
        makeup.append({"name": "foundation", "color": rgb_to_bgr(hex_to_rgb(foundation_colour))})

    if ('LIPS' in makeup_region):
        lipstick_colour = app.sidebar.color_picker(
        "LIPSTICK COLOUR", value="#0fffa0", key="LI")
        makeup.append({"name": "lips", "color": rgb_to_bgr(hex_to_rgb(lipstick_colour))})

    apply_button = app.sidebar.button('Apply Makeup', disabled=False)
    
    app.markdown("<h3 style='font-family: Inter; color:#160926; font-size: 24px; font-weight: 700; text-transform: uppercase'>RENDERING FEED</h3>",
                    unsafe_allow_html=True) 
    
   

app.sidebar.markdown('---')

# detection_confidence = app.sidebar.slider("Minimum confidence", key="dc", min_value=0.0, max_value=1.0, value=0.5, label_visibility="hidden")
# tracking_confidence = app.sidebar.slider("Minimum confidence", key="tc", min_value=0.0, max_value=1.0, value=0.5, label_visibility="hidden")


app.set_option('deprecation.showfileUploaderEncoding',False)

# record = app.sidebar.button("Record")

vid2Frame = app.empty();

vid2 = cv2.VideoCapture(0)

width2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_input2 = int(vid2.get(cv2.CAP_PROP_FPS))

# app.markdown("**Frame Rate (FPS)**")

# app.write(f"<h1 style='text-align: center; color: blue;'>{fps_input2}</h1>", unsafe_allow_html=True)


# recording
codec2 = cv2.VideoWriter_fourcc('V', 'P', '0', '9') # type: ignore
out2 = cv2.VideoWriter('out2.mp4', codec2, fps_input2, (width2, height2))
    
fps2 = 0
i2 = 0
if apply_button:
        # app.write(makeup)
    app.success('Makeup Rendered', icon="✅")

    app.markdown("<hr />", unsafe_allow_html=True)

prevTime = 0
# Initialize variables
total_frames = 0
successful_frames = 0
threshold_fps = 15  # Set your desired threshold FPS value

while vid2.isOpened():
    i2+= 1
    try:
        ret_val, frame2 = vid2.read()
        frame2 = cv2.flip(frame2, 1)
        if ret_val:
            # cv2.imshow("Original", frame)

            features = [];
            if (apply_button):
                features = makeup;

            feat_applied = apply_all_makeup(frame2, True, features, False)
            # cv2.imshow("ARMakeup", feat_applied)
            
            if cv2.waitKey(1) == 27:
                break
        try:
            frame2 = cv2.resize(feat_applied, (0,0), fx=0.8, fy=0.8)
            frame2 = image_resize(image=frame2,width=640)
        except Exception:
            print("Failed to resize")
            pass
        
    except Exception:
        print("An error occurred")
        pass

    # FPS counter
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    # # Check if FPS exceeds the threshold
    # if fps > threshold_fps:
    #     successful_frames += 1
    
    # total_frames += 1

    # app.write(f"<h1 style='text-align: center; color: blue;'>{int(fps)}</h1>", unsafe_allow_html=True)

    # # Calculate percentage of successfully tracked frames
    # success_rate = (successful_frames / total_frames) * 100
    # print(f"Percentage of successfully tracked frames: {success_rate:.2f}%")
    # data = [successful_frames, total_frames,int(fps), round(success_rate, 2)]

    # # Open the CSV file in append mode and create a CSV writer object
    # filename = 'data-movement.csv'

    # with open(filename, 'a', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)

    #     # Write the data as a new row in the CSV file
    #     csvwriter.writerow(data)

    # # Close the file after writing
    # csvfile.close()

    vid2Frame.image(frame2, channels='BGR', use_column_width=True)


vid2.release()

out2.release()

