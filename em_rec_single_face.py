import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
image_path = 'TrainPositive01-1e343.jpgface4.bmp'
    #sys.argv[1]
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading modelsf
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading images
# rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')
# print(face_coordinates)
# rgb_face = rgb_image
gray_face = gray_image

try:
    gray_face = cv2.resize(gray_face, (emotion_target_size))
except:
    print('exception ignored')

# rgb_face = preprocess_input(rgb_face, False)
# rgb_face = np.expand_dims(rgb_face, 0)

gray_face = preprocess_input(gray_face, True)
gray_face = np.expand_dims(gray_face, 0)
gray_face = np.expand_dims(gray_face, -1)
emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
#ИЗ ЭТОГО СЛОЯ ХОЧУ ВЫТАЩИТЬ ОЦЕНКИ АПОСТЕРИОРНЫХ ВЕРОЯТНОСТЕЙ
print(emotion_classifier.layers[44].output)
emotion_text = emotion_labels[emotion_label_arg]
print (emotion_text)
# bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
# cv2.imwrite('../images/predicted_test_image.png', bgr_image)
