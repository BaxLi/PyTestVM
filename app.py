import streamlit as st
from PIL import Image
from segment import process

st.title('Airplane segmentation demo')

# Добавление загрузчика файлов
image_file = st.file_uploader('Load an image', type=['png', 'jpg'])

# Выполнение блока, если загружено изображение
if not image_file is None:
    # Создание 2 колонок # st.beta_columns(2)
    col1, col2 = st.columns(2)
    # Открытие изображения
    image = Image.open(image_file)
    # Обработка изображения с помощью функции, реализованной в другом файле
    results = process(image_file)
    col1.text('Source image')
    # Вывод в первой колонке уменьшенного исходного изображения
    col1.image(results[0])
    col2.text('Mask')
    # Вывод маски второй колонке
    col2.image(results[1])
    # Вывод исходного изображения с наложенной маской (по центру)
    st.image(results[2])
