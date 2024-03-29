import os.path

import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image

from functions.classification import predict_single
from functions.segmentation import predict_mask_and_write
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches

# Place tensors on the CPU

my_path = os.path.abspath(os.path.dirname(__file__))
img_path = os.path.join(my_path, "../output/img.jpeg")
original_path = os.path.join(my_path, "../output/original.jpeg")
segmented_mask_path = os.path.join(my_path, "../output/mask.jpeg")
lung_extracted_path = os.path.join(my_path, "../output/lung_extracted.jpeg")
color_bar = os.path.join(my_path, "../output/color.png")


def load_image_and_save(image_file, path):
    img = Image.open(image_file)

    img.save(path)
    return img


def load_image(image_file):
    img = Image.open(image_file)
    return img


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def start():
    st.title("Chest Xray Analysis for Pneumonia and COVID-19")

    st.subheader("Image")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
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

        covid_prob = "{:.2%}".format(prediction_probability["Covid"][0])
        pneu_prob = "{:.2%}".format(prediction_probability["Pneumonia"][0])
        normal_prob = "{:.2%}".format(prediction_probability["Normal"][0])

        col1.metric("Covid-19", covid_prob)
        col2.metric("Pneumonia", pneu_prob)
        col3.metric("Normal", normal_prob)

        # st.table(prediction_probability)
        view_colormap(prediction_probability)
        st.image(load_image(color_bar))


def view_colormap(prediction_probability):
    fig, ax = plt.subplots(figsize=(8, 1))
    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.5, right=0.7)

    # cmap = mpl.cm.jet
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=0, vmax=100)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='horizontal', label='Accuracy')

    plt.axvline(x=prediction_probability["Covid"][0] * 100, color="red")
    plt.axvline(x=prediction_probability["Pneumonia"][0] * 100, color="green")
    plt.axvline(x=prediction_probability["Normal"][0] * 100, color="black")

    red_patch = mpatches.Patch(color='red', label='Covid-19 probability')
    green_patch = mpatches.Patch(color='green', label='Pneumonia probability')
    black_patch = mpatches.Patch(color='black', label='Normal probability')
    plt.legend(handles=[red_patch, green_patch, black_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    plt.savefig(color_bar)

