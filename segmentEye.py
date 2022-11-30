from PIL import Image
import numpy as np
from keras.models import load_model
MODEL_NAME = 'model_fmr_all.h5'
model = load_model(MODEL_NAME)  # Загружаем веса
INPUT_SHAPE = (28, 28, 1)


def process(image_file):
    image = Image.open(image_file).convert(
        'L')  # Открываем обрабатываемый файл
    # Изменяем размер изображения в соответствии со входом сети
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    # Меняем размерность тензора для подачи в сеть
    array = np.array(resized_image, dtype='float64') / 255
    array = array.reshape(-1, 28, 28, 1)
    cls_image = np.argmax(model.predict(array))

    return cls_image
