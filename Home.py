import streamlit as st

st.set_page_config(layout="wide",page_title="Curve Digitalization",page_icon="📈",)

col1, col2 = st.columns((1, 10))
col1.image('http://thepetro.cloud/wp-content/uploads/2023/05/Petrocloud.png',width=100)
col2.markdown("<h1 style='text-align: left'>Line Curve Digitalization</h1>", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 2px solid #4f8bf9;'/>", unsafe_allow_html=True)

css = """
        <style>
        input[type="number"] {
            text-align: center;
        }
        </style>
        """

st.markdown(css, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #4f8bf9;'>Functionality Description</h2>", unsafe_allow_html=True)
st.markdown("""<p style='text-align: justify;'>The algorithm combines DeepLabV3+ image segmentation architecture with k-means clustering to filter and differentiate between curves in the digital image. The ranges of the axes are used for interpolation.</p>""", unsafe_allow_html=True)
st.markdown("""
        <ul>
        <li style="font-size: 20px;"><strong>Input<strong>: Images, Axes-Range</li>
        <li style="font-size: 20px;"><strong>Output<strong>: Chosen data points containing the digitized curve in CSV format</li>
        </ul>
        """, unsafe_allow_html=True)
st.markdown("""
            <blockquote style="font-size: 18px;">
                <i>The user will need to manually select the relevant predicted figures.</i>
            </blockquote>
            """, unsafe_allow_html=True)
