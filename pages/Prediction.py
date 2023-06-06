import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2
from PIL import Image   
import dataiku
import os

st.markdown("<h2 style='text-align: center;'>Output Result 📝</h2>", unsafe_allow_html=True)
# Get the outputs variable from session state
outputs = st.session_state['outputs']

# Calculate the number of rows needed to display all images
num_rows = (len(outputs) + 1) // 2

# Loop over each row
for row in range(num_rows):
    # Create a new row using columns
    cols = st.columns(2)
    
    # Display the first image in this row (if it exists)
    index = row * 2
    if index < len(outputs):
        cols[0].header(f"Image {index + 1}")
        cols[0].image(outputs[index])
    
    # Display the second image in this row (if it exists)
    index += 1
    if index < len(outputs):
        cols[1].header(f"Image {index + 1}")
        cols[1].image(outputs[index])
