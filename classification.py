from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import numpy as np
import os

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

my_path = os.path.abspath(os.path.dirname(__file__))
mobileNet_model_path = os.path.join(my_path, "./models/MobileNetV2_model.h5")
resNet_model_path = os.path.join(my_path, "./models/ResNet50_model.h5")
xception_model_path = os.path.join(my_path, "./models/Xception_model.h5")
inceptionV3_model_path = os.path.join(my_path, "./models/InceptionV3_model.h5")


def predict_single(image):
    image = [image]
    # todo: change to ensemble
    model = load_model(mobileNet_model_path)
    batch_size = 1
    nb_samples = 1

    df_file = pd.DataFrame({
        'filename': image
    })

    test_single_gen = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df_file,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    predict = model.predict(test_single_gen, steps=np.ceil(nb_samples / batch_size))

    df_file['category'] = np.argmax(predict, axis=-1)
    label_map = {0: 'Covid', 1: 'Normal', 2: 'Pneumonia'}
    df_file['category'] = df_file['category'].replace(label_map)
    # print(predict[0])

    labels = ['Covid', 'Normal', 'Pneumonia']
    # probability.append( predict[0])
    #
    # pd_prob = pd.DataFrame(probability)
    df = pd.DataFrame(
        data = predict,
        columns=labels)
    # return df_file['category'][0], predict[0]
    return df_file['category'][0], df
    # return str(df_file['category'][0])
