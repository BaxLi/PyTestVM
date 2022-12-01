import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from keras.utils import img_to_array, load_img

MODEL_NAME_polyakov = 'polyakov_unet5_1e3.h5'
MODEL_NAME = 'MODEL_WEIGHTS_Final_Baxan_PSP_std_color_256x256_Real_Augment.h5'
model = load_model(MODEL_NAME)  # Загружаем веса
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 1)

# Цвета пикселов сегментированных изображений
Zracek = (255, 0, 0)         # Зрачек (красный)
Radujka = (0, 255, 0)           # Радужка (зеленый)
Sklera = (0, 0, 255)              # Склера (синий)
Other = (0, 0, 0)          # Окружение глаза ? ()

# список меток классов:
CLASS_LABELS = (Other, Zracek, Radujka, Sklera)
# Функция преобразования тензора меток класса в цветное сегметрированное изображение


def labels_to_rgb(image_list  # список одноканальных изображений
                  ):
    result = []
    # Для всех картинок в списке:
    for y in image_list:
        # Создание пустой цветной картики
        temp = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype='uint8')
        # По всем классам:
        for i, cl in enumerate(CLASS_LABELS):
            # Нахождение пикселов класса и заполнение цветом из CLASS_LABELS[i]
            temp[np.where(np.all(y == i, axis=-1))] = CLASS_LABELS[i]
        result.append(temp)
    return np.array(result)


def process_PSP_base(image_file):
    image = Image.open(image_file)  # Открытие обрабатываемого файла
    # Изменение размера изображения в соответствии со входом сети
    resized_image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    # Регулировка формы тензора для подачи в сеть
    # Запуск предсказания сети
    # Вычисление предсказания сети для картинок с отобранными индексами
    array = np.array(resized_image)[..., :3][np.newaxis, ..., np.newaxis]
    prediction_array = np.argmax(model.predict(array), axis=-1)
    # Подготовка цветов классов для отрисовки предсказания
    prediction_array = labels_to_rgb(prediction_array[..., None])

    return resized_image, prediction_array,


def process_polyacov(image_file, col):
    prediction_array_grey = None
    # image = type=<class 'PIL.PngImagePlugin.PngImageFile'>
    image = Image.open(image_file)
    # image_gr type=<class 'PIL.Image.Image'>
    image_gr = ImageOps.grayscale(image.resize((IMG_WIDTH, IMG_HEIGHT)))

    # col.text(f'st.image type={type(image)}')
    # col.text(f'st.image_GR!!! type={type(image_gr)}')
    # Загружаем изображение для функции предсказания.
    # img0 = image.load_img(im, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))

    # image_grX shape = (256, 256, 1)
    image_grX = img_to_array(image_gr)
    # col.text(f'image_grX shape = {image_grX.shape}')
    # image_grX = np.squeeze(image_grX, axis=-1)
    # col.text(f'image_grX_2 shape = {image_grX.shape}')
    # Переводим в нампи и нормализуем.
    image_grX = np.array([image_grX])
    prediction_array_grey = np.array(image_grX)/255
    # col.text(f'grX.shape={type(image_grX)}')
    # col.text(f'{image_grX}')
    # predict = np.argmax(
    #     model.predict([prediction_array_grey]), axis=-1)
    # Подготовка цветов классов для отрисовки предсказания
    # prediction_array_grey = labels_to_rgb(predict[..., None])
    return image_gr, prediction_array_grey
