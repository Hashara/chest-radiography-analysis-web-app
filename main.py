import streamlit as st
from PIL import Image
from segmentation import show_predictions, predict_mask_and_write
import numpy as np
import os.path
import tensorflow as tf

# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

my_path = os.path.abspath(os.path.dirname(__file__))
img_path = os.path.join(my_path, "img.jpeg")
original_path = os.path.join(my_path, "original.jpeg")


def load_image(image_file):
    print(image_file)
    img = Image.open(image_file)

    img.save(img_path)
    return img


def print_hi():
    st.title("Hello")

    st.subheader("Image")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        # To See details
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        st.write(file_details)

        # To View Uploaded Image
        st.image(load_image(image_file))
        predict_mask_and_write(img_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()
    # predict_mask_and_write(original_path)
