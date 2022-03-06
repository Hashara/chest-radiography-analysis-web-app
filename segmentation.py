import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os.path

H, W = 224, 224
input_shape = (H, W, 3)
my_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(my_path, "./models/unet_model.hdf5")
segmented_mask_path = os.path.join(my_path, "mask.jpeg")
preprocessed_img_path = os.path.join(my_path, "preprocessed.jpeg")
lung_extracted_path = os.path.join(my_path, "lung_extracted.jpeg")


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


smooth = 1e-15


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


model = tf.keras.models.load_model(model_path,
                                   custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef, "iou": iou})


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x= path
    x = cv2.resize(x, (W, H))

    # Apply CLAHE
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    x = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    x = x / 255.0
    x = x.astype(np.float32)
    return x


def create_mask(pred_mask):
    pred_mask = pred_mask.astype(np.int32)
    pred_mask = np.concatenate([pred_mask, pred_mask, pred_mask], axis=-1)
    return pred_mask * 255


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        fig, ax = plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    # st.pyplot()


def show_predictions(image):
    image = read_image(image)
    pred_mask = model.predict_single(image) > 0.5
    # print(pred_mask)
    display([image[0], create_mask(pred_mask)])
    # cv2.imwrite(segmented_mask_path, pred_mask)


def predict_mask_and_write(image):
    x = image
    """ Extracing the image name. """
    # image_name = x.split("/")[-1]

    """ Reading the image """
    ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
    ori_x = cv2.resize(ori_x, (W, H))

    # Apply CLAHE
    lab = cv2.cvtColor(ori_x, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    ori_x = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    x = ori_x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    """ Predicting the mask. """
    y_pred = model.predict(x)[0] > 0.5

    # replaced_image = replace_pixels(ori_x, y_pred)
    # write image
    y_pred = apply_morphology(y_pred)

    cv2.imwrite(segmented_mask_path, y_pred * 255)
    cv2.imwrite(preprocessed_img_path, ori_x)
    extracted = replace_pixels(image=cv2.imread(preprocessed_img_path), mask=cv2.imread(segmented_mask_path))
    cv2.imwrite(lung_extracted_path, extracted)


def apply_morphology(binary_img):
    dl_img = ndimage.binary_dilation(binary_img, iterations=8, border_value=0, brute_force=False)

    er_img = ndimage.binary_erosion(dl_img, iterations=6, border_value=1).astype(np.float32)

    return er_img


def replace_pixels(image, mask):
    out = mask.copy()
    out[mask == 255] = image[mask == 255]
    # cv2_imshow(out)
    return out
