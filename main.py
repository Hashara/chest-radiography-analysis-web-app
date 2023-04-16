import tensorflow as tf
import streamlit as st

from pages.classification_page import start, local_css
from pages.detail_page import start_detail_page

PAGES = [
    'Respiratory Diseases',
    'Classify Chest Xray'
]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with tf.device('/CPU:0'):
        url_params = st.experimental_get_query_params()
        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                st.experimental_set_query_params(page='Respiratory Diseases')
                url_params = st.experimental_get_query_params()

            st.session_state.page = PAGES.index(url_params['page'][0])
            st.session_state['loaded'] = False

        st.sidebar.title('Chest Xray Analysis')


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
        elif page == 'Classify Chest Xray':
            st.session_state.page = 1
            local_css("style.css")
            start()


        # with tf.device('/CPU:0'):
        #     local_css("style.css")
        #     start()
