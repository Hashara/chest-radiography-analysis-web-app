import tensorflow as tf
import streamlit as st

import pages

from pages.classification_page import start, local_css
from pages.detail_page import start_detail_page
from pages.methodology_page import start_methodology_page
from pages.results_page import start_result_page

PAGES = [
    'Respiratory Diseases',
    'Methodology',
    'Results',
    'Classify Chest Xray'
]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with tf.device('/CPU:0'):
        st.set_page_config(layout="wide", page_title="ChestXrayAnalysis")

        no_sidebar_style = """
            <style>
                div[data-testid="stSidebarNav"] {display: none;}
            </style>
        """
        st.markdown(no_sidebar_style, unsafe_allow_html=True)

        url_params = st.experimental_get_query_params()
        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                st.experimental_set_query_params(page='Respiratory Diseases')
                url_params = st.experimental_get_query_params()

            st.session_state.page = PAGES.index(url_params['page'][0])
            st.session_state['loaded'] = False

        st.sidebar.title('Chest Xray Analysis')

        st.sidebar.write("""
                An automated Radiography analysis framework for Pneumonia and Covid-19 identification
                can be used to provide better performance in chest x-ray analysis for detecting lung 
                infection conditions.
                """)
        # selection = st.sidebar.radio("Go to", PAGES)

        if st.session_state.page:
            page = st.sidebar.radio('Navigation', PAGES, index=st.session_state.page)
        else:
            page = st.sidebar.radio('Navigation', PAGES, index=0)

        st.experimental_set_query_params(page=page)

        # case switch for pages
        if page == 'Respiratory Diseases':
            st.session_state.page = 0
            start_detail_page()

        elif page == 'Methodology':
            st.session_state.page = 1
            start_methodology_page()

        elif page == 'Results':
            st.session_state.page = 2
            start_result_page()

        elif page == 'Classify Chest Xray':
            st.session_state.page = 3
            local_css("style.css")
            start()



        # with tf.device('/CPU:0'):
        #     local_css("style.css")
        #     start()
