import tensorflow as tf

from pages.classification_page import start, local_css

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with tf.device('/CPU:0'):
        local_css("style.css")
        start()
