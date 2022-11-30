import streamlit as st
from PIL import Image
from segment import process

st.title("EYE'S segmentation demo")
results = []
# Добавление загрузчика файлов
image_file = st.file_uploader(
    'Load an eye image (PNG or JPG)', type=['png', 'jpg'])

# Выполнение блока, если загружено изображение

if not image_file is None:                    # Выполнение блока, если загружено изображение
    # Создание 2 колонок # st.beta_columns(2)
    col1, col2 = st.columns(2)
    image = Image.open(image_file)            # Открытие изображения
    # Обработка изображения с помощью функции, реализованной в другом файле
    results = process(image_file)
    col1.text('Original image')
    # Вывод в первой колонке уменьшенного исходного изображения
    col1.image(results[0])
    col1.text(
        'SEGMENTATION result \n MODEL_WEIGHTS_Final_Baxan_PSP_std_color_256x256_Real_Augment')
    col1.image(results[1])                 # Вывод маски второй колонке
    col2.text('1')
    col2.text('2')
    col2.text('Grayscale case - Model')
    col2.image(results[2])
