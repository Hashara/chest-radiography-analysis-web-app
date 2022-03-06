import streamlit as st
from PIL import Image
from segmentation import show_predictions, predict_mask_and_write

from classification import predict_single
import numpy as np
import os.path
import tensorflow as tf

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
    st.title("Chest Xray Analysis")

    st.subheader("Image")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        # To See details
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        st.write(file_details)

        # To View Uploaded Image
        st.image(load_image_and_save(image_file, img_path), width=100)  # load and save uploaded image
        predict_mask_and_write(img_path)  # segmentation and lung extraction
        st.image(load_image(lung_extracted_path))

        prediction, prediction_probability = predict_single(lung_extracted_path)

        st.subheader('Prediction')
        st.write(prediction)

        st.subheader('Prediction Probability')

        st.table(prediction_probability)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with tf.device('/CPU:0'):
        print_hi()
        # predict_single(lung_extracted_path)
