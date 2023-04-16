import streamlit as st
from PIL import Image
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))


def start_methodology_page():
    st.title("Methodology")

    st.write("""
    Respiratory diseases have been known to be a main cause of death worldwide. Pneumonia 
    and Covid-19 are two of such dominant diseases. Since both diseases can lead
    to life-threatening conditions, detecting these conditions at an early stage is crucial to
    properly treat the patients. While chest X-rays are widely used for diagnosing these
    diseases, it requires expert knowledge. Several deep learning based studies are available
    in the literature that classify infection conditions in chest X-ray images. In addition,
    image segmentation has been also applied to obtain promising results in deep
    learning approaches. Our research focuses on using a modified version of the U-Net 
    architecture to conduct segmentation on chest X-rays and then use segmented images for
    classification which is conducted using an ensemble model developed using modified
    architectures of MobileNetV2, Resnet50, InceptionV3, and Xception. An accuracy of
    99.36% and a recall of 99.38% was achieved with the developed ensemble model.
    """)

    st.subheader("Dataset")
    st.markdown("""
    - [V7-labs COVID-19 X-ray dataset](https://github.com/v7labs/COVID-19-xray-dataset)
    - [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 
    - [Covid19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
    """)

    st.subheader("Process flow")
    st.image(Image.open(os.path.join(my_path, "../assets/images/process_flow.png")))
    st.write("""
    The process flow, as shown in above, mainly consists of segmentation and 
    classification. First, lung segmentation is performed by the modified U-Net architecture.
    As the next step, morphological image dilation and erosion techniques are applied on
    the segmented masks and afterwards the lung areas in the masks are extracted. Then
    the extracted lung images are used to train the modified MobileNetV2, InceptionV3,
    ResNet50 and Xception models with added top layers. To achieve even better results
    an ensemble is created by combining the four models.
    """)


