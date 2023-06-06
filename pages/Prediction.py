import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import cv2
from PIL import Image   
import dataiku
import os

st.markdown("<h2 style='text-align: center;'>Output Result 📝</h2>", unsafe_allow_html=True)

