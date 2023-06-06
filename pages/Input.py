import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2
from PIL import Image   
import dataiku
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#Preprocessing Lib Start

def random_color(C):
    # C is an integer that represents the number of colors to output
    # The function returns a list of C random and distinct RGB colors
    # The colors are not too white (i.e. the sum of RGB values is less than 600)
    colors = [] # an empty list to store the colors

    while len(colors) < C: # loop until we have C colors
        r = np.random.randint(0, 255) # generate a random red value
        g = np.random.randint(0, 255) # generate a random green value
        b = np.random.randint(0, 255) # generate a random blue value
        color = (r, g, b) # create a tuple for the color
        if sum(color) < 600 and color not in colors: # check if the color is not too white and not already in the list
            colors.append(color) # add the color to the list
    return colors # return the list of colors

def rgb_to_hex(rgb):
    """Converts a tuple of RGB values to a hexadecimal color code.

    Parameters:
        rgb (tuple): A tuple of three integers in the range 0-255 representing the red, green and blue components of the color.

    Returns:
        str: A string of six hexadecimal digits representing the color code.

    Example:
        >>> rgb_to_hex((255, 0, 0))
        'FF0000'
    """
    # Check if the input is a valid RGB tuple
    if not isinstance(rgb, tuple) or len(rgb) != 3 or not all(0 <= x <= 255 for x in rgb):
        raise ValueError("Invalid RGB input")

    # Convert each component to a two-digit hexadecimal string
    hex_components = [format(x, "02X") for x in rgb]

    # Join the components and return the result
    return "".join(hex_components)

def generate_tuples(N):
    result = []
    for i in range(N):
        result.append(((255-i)/255, 0, 0))
    return result

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def resize(img,k):
    return cv2.resize(img, dsize=(int(img.shape[1]*k), int(img.shape[0]*k)), interpolation=cv2.INTER_NEAREST)


def crop_image_v2(image, crop_size, pixel_value):
  # Get the shape of the image
  height, width, N = image.shape
  # Get the crop height and width
  crop_height, crop_width = crop_size
  # Calculate the number of crops along each dimension
  num_crops_h = height // crop_height + (height % crop_height != 0)
  num_crops_w = width // crop_width + (width % crop_width != 0)
  # Calculate the total number of crops
  num_crops = num_crops_h * num_crops_w
  # Initialize the output array with the padding value
  output = np.full((num_crops, crop_height, crop_width, N), pixel_value)
  # Loop over the crops and copy the image data
  for i in range(num_crops_h):
    for j in range(num_crops_w):
      # Get the index of the current crop
      k = i * num_crops_w + j
      # Get the start and end coordinates of the image region to copy
      start_y = i * crop_height
      end_y = min((i + 1) * crop_height, height)
      start_x = j * crop_width
      end_x = min((j + 1) * crop_width, width)
      # Get the start and end coordinates of the output region to paste
      out_start_y = 0
      out_end_y = end_y - start_y
      out_start_x = 0
      out_end_x = end_x - start_x
      # Copy the image data to the output array
      output[k, out_start_y:out_end_y, out_start_x:out_end_x, :] = image[start_y:end_y, start_x:end_x, :]
  # Return the output array and the number of crops along each dimension
  return output, (num_crops_h, num_crops_w)

def uncrop_image_v2(cropped_images, num_crops):
  # Get the shape of the cropped images
  num_crops_total, crop_height, crop_width, N = cropped_images.shape
  # Get the number of crops along each dimension
  num_crops_h, num_crops_w = num_crops
  # Calculate the height and width of the original image
  height = num_crops_h * crop_height
  width = num_crops_w * crop_width
  # Initialize the output array with zeros
  output = np.zeros((height, width, N))
  # Loop over the cropped images and copy them to the output array
  for i in range(num_crops_h):
    for j in range(num_crops_w):
      # Get the index of the current cropped image
      k = i * num_crops_w + j
      # Get the start and end coordinates of the output region to paste
      start_y = i * crop_height
      end_y = (i + 1) * crop_height
      start_x = j * crop_width
      end_x = (j + 1) * crop_width
      # Copy the cropped image data to the output array
      output[start_y:end_y, start_x:end_x, :] = cropped_images[k]
  # Return the output array as an integer type (to match the original image type)
  return output.astype(np.uint8)

def predict_curves(img,model):
    crop_sizes = 32*10
    c_img, ri = crop_image_v2(img,(crop_sizes,crop_sizes),255)
    p = []
    for i in range(c_img.shape[0]):
        p.append(model.predict(c_img[i:i+1]/255.0))
    p = np.concatenate(p)

    p[p <= 0.9] = 0
    p[p > 0.9] = 1
    re_mask = uncrop_image_v2(p,ri)
    re_img = uncrop_image_v2(c_img,ri)
    
    return re_img, re_mask


def mask_flattened(c_msk):
    # Create an empty array with the same shape as the first three dimensions of c_msk
    result = np.zeros(c_msk.shape[:-1])
    
    # Set the value of result to 1 where any layer of c_msk is active
    result[np.any(c_msk > 0, axis=-1)] = 1
    
    return np.expand_dims(result,len(result.shape))

def predict_colors(re_img,re_msk,N):

    # Load and reshape the image
    image = re_img * mask_flattened(re_msk)
    image[image == 255] = 0
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # Select only the pixels that are non-zero on all three channels
    non_zero_pixels = non_zero_pixels = pixel_vals[np.nonzero(np.all(pixel_vals != 0, axis=-1) & np.any(pixel_vals != 255, axis=-1))]

    # Apply k-means clustering with k=3 on the non-zero pixels
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(non_zero_pixels)
    centers = kmeans.cluster_centers_
    centers = np.uint8(centers)


    # Print the RGB values of the three most dominant colors
    print("The RGB values of the three most dominant colors are:")
    for i in range(N):
        print(f"Cluster {i}: {centers[i]}")
    
    centers_hex = ['#'+str(rgb_to_hex(tuple(i))) for i in centers]
    
    return centers

def black_to_white(image: np.ndarray) -> np.ndarray:
    # Create a mask for black pixels
    mask = np.all(image == [0, 0, 0], axis=-1)
    
    # Change black pixels to white
    image[mask] = [255, 255, 255]
    
    return image

def get_n_most_common_colors(image_a, n):
    img = Image.fromarray(image_a)
    colors = img.getcolors(img.size[0] * img.size[1])
    sorted_colors = sorted(colors, key=lambda t: t[0], reverse=True)
    return [np.array(color[1]) for color in sorted_colors[:n]]

def predict_mask(img,mask,N):
    centers = get_n_most_common_colors(cv2.bitwise_and(img, img, mask),N)
    threshold = 32
    outputs = []
    for i in range(len(tuple(centers))):
        rgb_color = tuple(centers[i])
        lower = np.array([max(0, c - threshold) for c in rgb_color])
        upper = np.array([min(255, c + threshold) for c in rgb_color])
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        output = black_to_white(output)
        outputs.append(output)
    return outputs, centers


def analyze_mask(mask: np.ndarray, stat: str = 'median', n_v: int = 2) -> np.ndarray:
    """
    Analyze a mask image and return the mean or median position of pixels with value 1 in each row.

    :param mask: A 2D numpy array representing the mask image.
    :param stat: The statistical measure to use ('mean' or 'median').
    :param n_v: The number of neighbors to check in each direction (up and down).
    :return: A tuple containing two 1D numpy arrays representing the filtered X and Y values.
    """
    assert stat in ['mean', 'median'], "Invalid value for 'stat'. Must be 'mean' or 'median'."
    result = []
    for row in mask:
        positions = np.where(row == 1)[0]
        if len(positions) > 0:
            if stat == 'mean':
                result.append(np.mean(positions))
            else:
                result.append(np.median(positions))
        else:
            result.append(mask.shape[1]+1)
    results = zip(range(len(result)),result)
    filtered = []
    for i in results:
        if (i[1] < mask.shape[1]):
            filtered.append(i)
    filtered = np.array(filtered)
    Y = filtered[:,0]
    X = filtered[:,1]
    
    # Additional feature
    X_filtered = []
    Y_filtered = []
    for i in range(len(X)):
        start = max(0, i - n_v)
        end = min(len(X), i + n_v + 1)
        neighbors = X[start:end]
        q1 = np.percentile(neighbors, 25)
        q3 = np.percentile(neighbors, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        if lower_bound <= X[i] <= upper_bound:
            X_filtered.append(X[i])
            Y_filtered.append(Y[i])
    
    return (np.array(X_filtered), np.array(Y_filtered))


#Preprocessing Lib End

@st.cache_resource
def load_model():
    host = "https://ai-schlumberger-eag-consulting.p.datascience.cloud.slb-ds.com"
    apiKey = 'CvyAhYL5Y1AO5VLd1aNDUUVSrgXRBFjL'

    client = dataiku.set_remote_dss(host, apiKey)

    model_folder = dataiku.Folder("Module1",'MODULE1')
    model_path = model_folder.list_paths_in_partition()

    tmp_file_path = 'better_predictor_v5.h5'

    with open(tmp_file_path, "wb") as tmp_file:
        with model_folder.get_download_stream(model_path[0]) as model_weight_file:
            tmp_file.write(model_weight_file.read())

    model = tf.keras.models.load_model(tmp_file_path,compile=False)
    os.remove(tmp_file_path)
    return model


if 'outputs' in st.session_state:
    del st.session_state['outputs']
if 'centers' in st.session_state:
    del st.session_state['centers']

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


N = st.number_input("Number of Curves on the image", value=1)
N += 15
realtime_update = True
accuracy = 1
width = 800
height = 800

canvas_resized = False

if bg_image is not None:
    crop_sizes = 32*10
    image = Image.open(bg_image)
    c_img, ri = crop_image_v2(np.asarray(image),(crop_sizes,crop_sizes),255)
    image = Image.fromarray(uncrop_image_v2(c_img,ri).astype('uint8'), 'RGB')
    width_ori, height_ori = image.size
    width, height = image.size
    max_length = 800
    if height > max_length:
        ratio = max_length / float(height)
        width = int(ratio * width)
        height = max_length
        image = image.resize((width, height), Image.LANCZOS)
        canvas_resized = True

# Add sliders to control the positions of the horizontal and vertical lines
y_axis_scale = st.sidebar.selectbox("Y-Axis Scale", ["normal", "log"])

st.sidebar.markdown("<b><span style='color:green'>Y-min (%):</span></b>", unsafe_allow_html=True)
h_line_min_position = st.sidebar.slider("", 0, 100, 75,accuracy,key="ymin")

st.sidebar.markdown("<b><span style='color:blue'>Y-max (%):</span></b>", unsafe_allow_html=True)
h_line_max_position = st.sidebar.slider("", 0, 100, 25,accuracy,key="ymax")

x_axis_scale = st.sidebar.selectbox("X-Axis Scale", ["normal", "log"])

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
    height=height,
    width=width,
)



# Define the predict_button variable before it is used
predict_button = False
save_button = st.button('Save line positions')
reset_button = st.button('Reset line positions')
if save_button:
    df = pd.DataFrame({
        'h_line_min_y': [h_line_min_y],
        'h_line_max_y': [h_line_max_y],
        'v_line_min_x': [v_line_min_x],
        'v_line_max_x': [v_line_max_x],
        'image_width': [image.size[0]] if bg_image else [None],
        'image_height': [image.size[1]] if bg_image else [None],
        'image_width_ori': [width_ori],
        'image_height_ori': [height_ori],
        'y_axis_scale': [y_axis_scale],
        'x_axis_scale': [x_axis_scale],
        'num_curves': [N]
    })

    st.success('Line positions and image dimensions saved as CSV file')
    
    # Save the image and dataframe to session state
    if bg_image is not None:
        image = Image.open(bg_image)
        image_data = np.asarray(image)
        st.session_state['image_data'] = image_data
    st.session_state['df'] = df

if reset_button:
    df = pd.DataFrame({
        'h_line_min_y': [0],
        'h_line_max_y': [0],
        'v_line_min_x': [0],
        'v_line_max_x': [0],
        'image_width': [None],
        'image_height': [None],
        'image_width_ori': [0],
        'image_height_ori': [0],
        'y_axis_scale': ["normal"],
        'x_axis_scale': ["normal"],
        'num_curves': [1]
    })
    

    st.success('Line positions and image dimensions reset to 0')
    
    # Remove the saved image and dataframe from session state
    if 'image_data' in st.session_state:
        del st.session_state['image_data']
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'outputs' in st.session_state:
        del st.session_state['outputs']
    if 'centers' in st.session_state:
        del st.session_state['centers']

# Create the Predict button outside of any conditional blocks
st.markdown("<h1 style='text-align: left;'>Curve Digitizer</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left;'></h2>", unsafe_allow_html=True)  
st.markdown("<h2 style='text-align: left;'>Predict Curves</h2>", unsafe_allow_html=True)
predict_button = st.button('load_model')

if predict_button:
    st.markdown("<h2 style='text-align: center;'>Output Result 📝</h2>", unsafe_allow_html=True)
    model = load_model()
    st.success('Model Loaded')
    model.summary(print_fn=lambda x: st.text(x))
    image = Image.open(bg_image)
    st.write(np.asarray(image).shape)
    re_img,re_mask = predict_curves(np.asarray(image),model)
    st.success('Model Loaded')
    N = 10
    image = re_img
    threshold = 32 # example threshold value
    outputs, centers = predict_mask(re_img,re_mask,10)
    st.session_state['outputs'] = outputs
    st.session_state['centers'] = centers
    n_focus = range(len(centers))

    focus = [outputs[i] for i in n_focus]
    focus = [mask_flattened(i) for i in focus]
    results = [analyze_mask(i,'median',10) for i in focus]

    # Iterate through the outputs and display each image using Streamlit
    for i in range(len(outputs)):
        # Convert the numpy array to a PIL.Image.Image object
        output_image = Image.fromarray(outputs[i])
    
        # Display the image using Streamlit
        st.image(output_image, caption=f"Output {i+1}")


    # Calculate the number of rows needed for the subplots
    N = len(results)
    num_rows = (N + 2) // 3

    # Create the subplots
    fig = make_subplots(rows=num_rows, cols=3)
    re_img_pil = Image.fromarray(re_img)

    # Iterate through the results and add a scatter plot for each curve
    for i in range(N):
        row = i // 3 + 1
        col = i % 3 + 1
        X = results[i][0]
        Y = results[i][1]
        fig.add_trace(
            go.Scatter(x=X, y=Y, mode='markers', marker=dict(size=5)),
            row=row, col=col
        )
        fig.add_layout_image(
            dict(
                source=re_img_pil,
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                opacity=0.5,
                layer="below"
            ),
            row=row, col=col
        )

    # Update the layout of the subplots
    fig.update_layout(
        height=num_rows * 400,
        showlegend=False
    )

    # Show the plot
    st.plotly_chart(fig)

    
