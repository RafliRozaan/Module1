import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os


# Specify canvas parameters in application
drawing_mode = 'line'

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 2)
h_line_color_1 = "blue"
h_line_color_2 = "green"
v_line_color_1 = "red"
v_line_color_2 = "black"
bg_color = "#eee"
st.markdown("<h2 style='text-align: left;'>Set the X and Y axis on the Figure</h2>", unsafe_allow_html=True)
bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = True
accuracy = 1
width = 800
height = 800

canvas_resized = False

if bg_image is not None:
    image = Image.open(bg_image)
    width, height = image.size
    max_length = 800
    if height > max_length:
        ratio = max_length / float(height)
        width = int(ratio * width)
        height = max_length
        image = image.resize((width, height), Image.ANTIALIAS)
        canvas_resized = True

# Add sliders to control the positions of the horizontal and vertical lines
st.sidebar.markdown("<b><span style='color:green'>Y-min (%):</span></b>", unsafe_allow_html=True)
h_line_min_position = st.sidebar.slider("", 0, 100, 75,accuracy,key="ymin")

st.sidebar.markdown("<b><span style='color:blue'>Y-max (%):</span></b>", unsafe_allow_html=True)
h_line_max_position = st.sidebar.slider("", 0, 100, 25,accuracy,key="ymax")

st.sidebar.markdown("<b><span style='color:red'>X-min (%):</span></b>", unsafe_allow_html=True)
v_line_min_position = st.sidebar.slider("", 0, 100, 25,accuracy,key="xmax")

st.sidebar.markdown("<b><span style='color:black'>X-max (%):</span></b>", unsafe_allow_html=True)
v_line_max_position = st.sidebar.slider("", 0, 100, 75,accuracy,key="xmin")

# Calculate the y-coordinates of the horizontal lines and the x-coordinates of the vertical lines based on the slider values
h_line_min_y = int(height * h_line_min_position / 100)
h_line_max_y = int(height * h_line_max_position / 100)
v_line_min_x = int(width * v_line_min_position / 100)
v_line_max_x = int(width * v_line_max_position / 100)

# Create a canvas component

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    background_color=bg_color,
    background_image=image if bg_image else None,
    update_streamlit=realtime_update,
    drawing_mode=drawing_mode,
    initial_drawing={
        "version": "4.4.0",
        "objects": [
            {
                "type": "line",
                "version": "4.4.0",
                "originX": "center",
                "originY": "center",
                "left": width / 2,
                "top": h_line_min_y,
                "width": width,
                "height": 0,
                "fill": h_line_color_2,
                "stroke": h_line_color_2,
                "strokeWidth": stroke_width,
                "x1": -width / 2,
                "x2": width / 2,
                "y1": 0,
                "y2": 0,
            },
            {
                "type": "line",
                "version": "4.4.0",
                "originX": "center",
                "originY": "center",
                "left": width / 2,
                "top": h_line_max_y,
                "width": width,
                "height": 0,
                "fill": h_line_color_1,
                "stroke": h_line_color_1,
                "strokeWidth": stroke_width,
                "x1": -width / 2,
                "x2": width / 2,
                "y1": 0,
                "y2": 0,
            },
            {
                "type": "line",
                "version": "4.4.0",
                "originX": "center",
                                "originY": "center",
                "left": v_line_min_x,
                "top": height / 2,
                "width": 0,
                "height": height,
                "fill": v_line_color_1,
                "stroke": v_line_color_1,
                "strokeWidth": stroke_width,
                "x1": 0,
                "x2": 0,
                "y1": -height / 2,
                "y2": height / 2,
            },
            {
                "type": "line",
                "version": "4.4.0",
                "originX": "center",
                "originY": "center",
                "left": v_line_max_x,
                "top": height / 2,
                "width": 0,
                "height": height,
                "fill": v_line_color_2,
                "stroke": v_line_color_2,
                "strokeWidth": stroke_width,
                "x1": 0,
                "x2": 0,
                "y1": -height / 2,
                "y2": height / 2,
            },
        ],
        "background": bg_color,
    },
    height=height if canvas_resized else None,
    width=width if canvas_resized else None,
)



# Add a button to save the line positions as a CSV file
if st.button('Save line positions'):
    df = pd.DataFrame({
        'h_line_min_y': [h_line_min_y],
        'h_line_max_y': [h_line_max_y],
        'v_line_min_x': [v_line_min_x],
        'v_line_max_x': [v_line_max_x],
        'image_width': [image.size[0]] if bg_image else [None],
        'image_height': [image.size[1]] if bg_image else [None],
        'body_width': [width],
        'body_height': [height]
    })
    #df.to_csv('line_positions.csv', index=False)
    st.success('Line positions and image dimensions saved as CSV file')
    
    # Save the uploaded image as prediction_target.jpg
    #if bg_image is not None:
    #    image = Image.open(bg_image)
    #    image.save('prediction_target.jpg')

if st.button('Reset line positions'):
    df = pd.DataFrame({
        'h_line_min_y': [0],
        'h_line_max_y': [0],
        'v_line_min_x': [0],
        'v_line_max_x': [0],
        'image_width': [None],
        'image_height': [None],
        'body_width': [0],
        'body_height': [0]
    })
    #df.to_csv('line_positions.csv', index=False)
    st.success('Line positions and image dimensions reset to 0')
    
    # Delete the prediction_target.jpg file
    if os.path.exists('prediction_target.jpg'):
        os.remove('prediction_target.jpg')




