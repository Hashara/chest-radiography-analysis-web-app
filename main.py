import os.path

import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

from classification import predict_single
from segmentation import predict_mask_and_write

# Place tensors on the CPU


my_path = os.path.abspath(os.path.dirname(__file__))
img_path = os.path.join(my_path, "img.jpeg")
original_path = os.path.join(my_path, "original.jpeg")
segmented_mask_path = os.path.join(my_path, "mask.jpeg")
lung_extracted_path = os.path.join(my_path, "lung_extracted.jpeg")


def load_image_and_save(image_file, path):
    img = Image.open(image_file)

    img.save(path)
    return img


def load_image(image_file):
    img = Image.open(image_file)
    return img


def print_hi():
    st.title("Chest Xray Analysis for Pneumonia and COVID-19")

    st.subheader("Image")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        # To See details
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        # st.write(file_details)

        col1, col2 = st.columns(2)
        #
        with col1:
            st.header("Input image")
            st.image(load_image_and_save(image_file, img_path), width=300)  # load and save uploaded image

        predict_mask_and_write(img_path)  # segmentation and lung extraction

        with col2:
            st.header("Lung extracted image")
            st.image(load_image(lung_extracted_path), width=300)

        prediction, prediction_probability = predict_single(lung_extracted_path)

        st.subheader('Prediction')
        # st.write(prediction)
        new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">' + prediction + '</p>'
        st.markdown(new_title, unsafe_allow_html=True)

        st.subheader('Prediction Probability')

        col1, col2, col3 = st.columns(3)
        col1.metric("Covid-19", prediction_probability["Covid"][0])
        col2.metric("Pneumonia", prediction_probability["Pneumonia"][0])
        col3.metric("Normal", prediction_probability["Normal"][0])

        st.table(prediction_probability)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with tf.device('/CPU:0'):
        print_hi()
        # predict_single(lung_extracted_path)
