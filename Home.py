import streamlit as st

st.set_page_config(layout="wide",page_title="Curve Digitalization",page_icon="📈",)

col1, col2 = st.columns((1,20))
col1.image('http://thepetro.cloud/wp-content/uploads/2023/05/Petrocloud.png',width=100)
col2.markdown("<h1 style='text-align: center;'>Line Curve Digitalization</h1>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

css = """
        <style>
        input[type="number"] {
            text-align: center;
        }
        </style>
        """

st.markdown(css, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Functionality Description</h2>", unsafe_allow_html=True)
st.markdown("""<h4 style='text-align: justify;'>The algorithm works by combining DeepLabV3+ images segmentation architecture to segmentize the digital curve by filtering the curves line that are potentially the data points and differentiating between curves using k-mean. 
                    The ranges of axes are needed for interpolation.</h4>""", unsafe_allow_html=True)
st.markdown("""
        <ul>
        <li style="font-size: 20px;"><strong>Input<strong>: Images, Axes-Range</li>
        <li style="font-size: 20px;"><strong>Output<strong>: Choosen data points containing the digitized curve in CSV format</li>
        </ul>
        """, unsafe_allow_html=True)
st.markdown("""
            <blockquote style="font-size: 18px;">
                <i>User will need to manually select the relevant predicted figures.</i>
            </blockquote>
            """, unsafe_allow_html=True)
