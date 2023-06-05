import streamlit as st
import numpy as np
import tensorflow as tf
from ProcessingLib import *

st.set_page_config(page_title='Streamlit App', layout='wide')

st.sidebar.title('Sidebar')
bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

st.markdown("<h1 style='text-align: center;'>Curve Digitizer</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left;'></h2>", unsafe_allow_html=True)  
st.markdown("<h2 style='text-align: left;'>Predict Curves</h2>", unsafe_allow_html=True)
predict_button = st.button('load_model')

if predict_button:
    model = tf.keras.models.load_model('better_predictor_v5.h5')

    st.success('Model Loaded')
    model.summary(print_fn=lambda x: st.text(x))
    image = Image.open(bg_image)
    st.write(np.asarray(image).shape)
    re_img,re_mask = predict_curves(np.asarray(image),model)
    st.success('Model Loaded')
    st.image(re_mask)
    N = 10
    image = re_img
    st.image(re_mask)
    threshold = 32 # example threshold value