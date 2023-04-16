import streamlit as st
from PIL import Image
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))


def start_result_page():
    st.title("Results")

    st.write("""
    The dice coefficient, intersection over union (IoU), precision and recall were used as
    the performance metrics to assess the performance of the U-Net model while accuracy,
    recall, precision, and F1 score were used to assess the performance of the classification
    models and the ensemble. It was observed that the modified U-Net model could obtain
    better results than the original U-Net model
    """)

    unet_result = [["Matric", "Modified U-Net", "Original U-Net"],
                   ["Dice coefficient %", "96.65", "94.97"],
                   ["IoU score %", "93.53", "90.45"],
                   ["Precision %", "97.35", "95.68"],
                   ["Recall %", "96.00", "95.05"]
                   ]

    st.subheader("U-Net model performance")
    st.write("""
    The could obtain better results than the original U-Net model. The modified U-Net 
    architecture has achieved higher results for the dice co-efficient, IoU score,
    precision, and recall. The dice coefficient and IoU score measure the 
    similarity of the output mask to the ground truth mask
    """)

    st.table(unet_result)

    st.subheader("Classification model performance")
    st.write(""" 
    We have used the modified MobileNetV2, InceptionV3, ResNet50 and Xception models with
    added top layers to classify the segmented images. The ensemble model was created by
    combining the four models. The ensemble model has achieved the highest accuracy, recall,
    precision, and F1 score. The accuracy, recall, precision, and F1 score are used to
    measure the performance of the classification models and the ensemble. 
    """)

    st.image(Image.open(os.path.join(my_path, "../assets/images/all-results.png")))
