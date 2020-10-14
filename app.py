import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps



st.title("Dog and Cat Classification")
st.text("Upload a Dog or Cat Image for image classification")

uploaded_file = st.file_uploader("Dog or Cat Image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Uploaded .', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.title("It's a Dog !")
    else:
        st.title("It's a Cat !")