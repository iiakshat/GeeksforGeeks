import streamlit as st
from tensorflow import image
from keras import models
import numpy as np
from PIL import Image
import pandas as pd

st.title("Rice Disease Classifier 🌾")

desc = pd.read_csv("files/description.csv")
model  = models.load_model("models/0.3/model.h5")

dis = list(desc.disease.values)

def image_classifier(inp):
    try:
        inp = image.resize(inp, (256,256))
        inp = np.expand_dims(inp,0)
        pred= model.predict(inp)
        return dis[np.argmax(pred)] , f"Confidence - {round(max(pred[0])*100,2)}%"
    except:
        return "Healthy", "Confidence - 0%"

def detail(pro):
    x = desc[desc["disease"]==pro]
    return list(x["hindi"])[0], list(x["desc"])[0], list(x["hndesc"])[0], list(x["pre"])[0], list(x["hnpre"])[0]


cho = st.file_uploader("Upload Image From Gallery", type=['png','jpg','jpeg','webp'])
img = ""

if cho is not None:
    img = Image.open(cho)

st.write("or")
if st.button("Open Camera"):
    cam = st.camera_input("Take image")
    if cam is not None:
        img = Image.open(cam)


if st.button("Detect"):
    col1,col2,col3 = st.columns(3)
    pro, conf = image_classifier(img)
    hin, des, hnd, pre, hnp = detail(pro)
    try:
        with col2:
            st.image(img)
            st.write("\n\n")
            st.header(pro)
            st.subheader(f"({hin})")
            st.subheader(conf)
            st.write("\n\n\n\n")

            st.subheader(f"Description :")
            st.write(des)
            st.write("\n\n")
            st.write(hnd)
            st.write("\n\n\n")

            st.subheader(f"Precautions :")
            st.write(pre)
            st.write("\n\n")
            st.write(hnp)
    except:
        with col2:
            st.subheader(":red[Enter Valid Input]")
