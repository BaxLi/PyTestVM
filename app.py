import streamlit as st
from PIL import Image
from segment import process_PSP_base, process_polyacov

st.title("EYE'S segmentation demo")
results = []
# Добавление загрузчика файлов
image_file = st.file_uploader(
    'Load an eye image (PNG or JPG)', type=['png', 'jpg'])

# Выполнение блока, если загружено изображение
results_polyakov = None
if not image_file is None:                    # Выполнение блока, если загружено изображение
    # Создание 2 колонок # st.beta_columns(2)
    col1, col2, col3 = st.columns(3)
    col2.header('Col-2')

    image = Image.open(image_file)            # Открытие изображения
    # Обработка изображения с помощью функции, реализованной в другом файле
    results = process_PSP_base(image_file)
    results_polyakov = process_polyacov(image_file, col2)
    with col1:
        st.header('Original image')
        st.text('PSP Color')
        # Вывод в первой колонке уменьшенного исходного изображения
        st.image(results[0])
        st.text(
            'SEGMENTATION result \n MODEL_WEIGHTS_Final_Baxan_PSP_std_color_256x256_Real_Augment')
        st.image(results[1])                 # Вывод маски второй колонке

    with col3:
        st.header('U-NET')
        st.text(' --- ')
        if not results_polyakov[0] == None:
            st.image(results_polyakov[0])
        st.text('Grayscale case - Model')
        st.image('none')
