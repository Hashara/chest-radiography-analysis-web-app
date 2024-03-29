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

# define model paths
mobileNet_model_path = os.path.join(my_path, "../models/MobileNetV2.h5")
inceptionV3_model_path = os.path.join(my_path, "../models/InceptionV3.h5")
resnet_model_path = os.path.join(my_path, "../models/ResNet50.h5")
xception_model_path = os.path.join(my_path, "../models/Xception.h5")


def predict_single(image):
    image = [image]

    # load models
    model1_mobilenet = load_model(mobileNet_model_path)
    model2_inception = load_model(inceptionV3_model_path)
    model3_resnet = load_model(resnet_model_path)
    model4_xception = load_model(xception_model_path)

    models = [model1_mobilenet, model2_inception, model3_resnet, model4_xception]

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

    preds = [model.predict(test_single_gen, steps=np.ceil(nb_samples / batch_size)) for model in models]
    preds = np.array(preds)
    # weights .5:.5:.1:.1
    weights = [5 / 11, 4 / 11, 1 / 11, 1 / 11]

    weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
    weighted_ensemble_prediction = np.argmax(weighted_preds, axis=-1)

    df_file['category'] = weighted_ensemble_prediction
    label_map = {0: 'Covid', 1: 'Normal', 2: 'Pneumonia'}
    df_file['category'] = df_file['category'].replace(label_map)

    labels = ['Covid', 'Normal', 'Pneumonia']

    df = pd.DataFrame(
        data=weighted_preds,
        columns=labels)

    del model1_mobilenet
    del model2_inception
    del model3_resnet
    del model4_xception

    return df_file['category'][0], df
