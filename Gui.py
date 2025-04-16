import streamlit  as st
import numpy as n
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
##TO BE REMOVED
from scipy.fftpack import dct, idct

st.set_page_config(layout="wide",page_title="Watermarking tool")

main_container = st.container()
bigcol,col5= st.columns([6,8],gap = "small")
big_data = pd.read_csv("/home/dalil/Documents/Projects/Py_Image_processing/Data/Eval.csv")








st.markdown("""
    <style>
        .stRadio > div {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            padding: 13px;
        }
    </style>
""", unsafe_allow_html=True)

secondary_container =st.container()

with st.container():

    with bigcol:
        col1,col2 = st.columns([2,2],gap ="small")
        with col1:
            Cover = st.file_uploader("Upload Cover Image")
            if Cover is not None:
                    img = Image.open(Cover)
                    st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            Watermark = st.file_uploader("Upload Watermark Payload")
            if Watermark is not None:
                    img = Image.open(Watermark)
                    st.image(img, caption="Uploaded Image", use_container_width=True)

        col7,col8= st.columns([15,6],gap = "small")

        with col7:
            transform = st.radio("Choose Transformation",["DCT","DWT","DFT","LSB"])
            with st.container():
                if transform =="DCT" and (Cover is None or Watermark is None):
                    st.slider("Choose Quality")

        with col5:
            if transform =="DCT":
                if Cover is not None:
                        if Watermark is not None:

                            img = Image.open(Cover)
                            st.image(img, caption="Uploaded Image", use_container_width=True)
            if transform =="DCT":
                if Cover is not None:
                    if Watermark is not None:
                        ig = px.bar(big_data,
                                     x="Image",
                                     y=["PSNR","NC","MSII"],
                                     barmode="group",
                                     title="Comparison of PSNR, MSII, and NC across Images")

                        st.plotly_chart(ig)






            if transform =="DWT":
                st.write(" i am DWT")




            if transform == "DFT":
                st.write("I AM DFT")




            if transform == "LSB":
                st.write("I AM LSB")
