import streamlit as st
from PIL import Image
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))


def start_detail_page():
    st.title("Respiratory Diseases ")
    st.write("""
            Respiratory diseases such as pneumonia is a common lung infection condition and
        Coronavirus disease (COVID-19) has become a life-threatening disease that emerged
        in later 2019 and has been impacted the entire world. Pneumonia is a fatal lower
        respiratory infection under the acute diseases category and has been reported to be a
        major cause of deaths around the world.
    """)

    st.subheader("Pneumonia")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
            Pneumonia is a lung infection that inflames the air sacs in one or both lungs. The air 
            sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and
            difficulty breathing. A variety of organisms, including bacteria, viruses and fungi,
            can cause pneumonia. diagnosed and treated early, Pneumonia can be managed well with the
            help of antibiotics, viral and fungal medicine and complications that can lead to death
            could also be safely avoided. The most common method of diagnosing Pneumonia is through
            chest x-rays while various other tests like blood tests can also be of help. For detecting
            Pneumonia, one needs to examine the x-ray images carefully which requires expert knowledge
            and can be a time-consuming task
        """)
    with col2:
        col2.image(Image.open( os.path.join(my_path, "../assets/images/pneumonia.jpg")), width=300)

    st.subheader("COVID-19")
    # wirte content in a one column and display image in another column
    col1, col2 = st.columns([1,2])
    with col1:
        col1.image(Image.open( os.path.join(my_path, "../assets/images/covid.webp")), width=300)
    with col2:
        st.write("""
        COVID-19 is a fatal infectious disease caused by the SARS-CoV-2 virus. Although
        most of the infected experience mild to moderate symptoms, some have to experience
        serious symptoms that requires thorough medical attention. Elderly people and
        the people with underlying medical conditions like diabetics, cardiovascular disease,
        chronic respiratory disease, or cancer are having a high chance of the serious illness.
        As per the World Health Organization (WHO), anyone at any age can be infected with
        COVID-19 and cause death. While RT-PCR which is an expensive method, is
        used for the diagnosis of COVID-19 at the early stages, chest X-rays and computer
        tomography (CT) scans are a better diagnosis option for the patients that show the
        Pneumonia conditions.
        """)


