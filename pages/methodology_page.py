import streamlit as st
from PIL import Image
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))


def start_methodology_page():
    st.title("Methodology")