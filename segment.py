import numpy as np
from PIL import Image
from keras.models import load_model

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


def process(image_file):
    image = Image.open(image_file)  # Открытие обрабатываемого файла
    # Изменение размера изображения в соответствии со входом сети
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    # Регулировка формы тензора для подачи в сеть
    array = np.array(resized_image)[..., :3][np.newaxis, ..., np.newaxis]
    # Запуск предсказания сети
    # Вычисление предсказания сети для картинок с отобранными индексами
    prediction_array = np.argmax(model.predict(array), axis=-1)
    prediction_array_grey = np.argmax(model.predict(array), axis=-1)
    # Подготовка цветов классов для отрисовки предсказания
    prediction_array = labels_to_rgb(prediction_array[..., None])
    # prediction_array_grey = labels_to_rgb(prediction_array_grey[..., None])

    # prediction_array = model.predict(array)
    # # Нулевой канал предсказания (значения 0 - самолет, 1 - фон)
    # prediction_array = np.split(prediction_array, 2, axis=-1)[0]
    # # Создание массива нулей
    # zeros = np.zeros_like(prediction_array)
    # # Создание массива единиц
    # ones = np.ones_like(prediction_array)
    # prediction_array_4d = np.concatenate([255 * (prediction_array > 100), zeros, zeros, 128 * ones], axis=3)[
    #     0].astype(np.uint8)  # Формирование тензора для наложения найденной маски
    # # Преобразование тензора в изображение и подгонка его размера к исходному
    # mask_image = Image.fromarray(prediction_array_4d).resize(image.size)
    # # Добавление маски на исходное изображение
    # image.paste(mask_image, (0, 0), mask_image)
    # Возврат исходного уменьшенного изображения, найденной маски и исходного изображения с наложенной маской
    return resized_image, prediction_array, prediction_array_grey
