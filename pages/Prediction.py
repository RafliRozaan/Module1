import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2
from PIL import Image   
import dataiku
import os

bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

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
st.write(np.asarray(Image.open(bg_image)))