import streamlit as st
from PIL import Image
from segment import process_PSP_base, process_Gray_PSP

st.title("EYE'S segmentation demo")
results = []
# Добавление загрузчика файлов
image_file = st.file_uploader(
    'Load an eye image (PNG or JPG)', type=['png', 'jpg'])

# Выполнение блока, если загружено изображение
results_grey_psp = None
if not image_file is None:                    # Выполнение блока, если загружено изображение
    # Создание 2 колонок # st.beta_columns(2)
    col1, col2, col3 = st.columns(3)
    col2.header('   -')
    col2.text('SEGMENTATION result')
    # col2.text(f'st.image type={type(image_file)}')
    image = Image.open(image_file)            # Открытие изображения
    # Обработка изображения с помощью функции, реализованной в другом файле
    results = process_PSP_base(image_file)
    results_grey_psp = process_Gray_PSP(image_file, col2)

    with col1:
        st.header('Original image')
        st.text('PSP Color-256')
        # Вывод в первой колонке уменьшенного исходного изображения
        st.image(results[0])
        st.text(
            'Final_Baxan_PSP_std_color_256x256_Real_Augment')
        st.image(results[1])                 # Вывод маски второй колонке

    with col3:
        st.header(' -')
        st.text('PSP-Grey-256')
        if not results_grey_psp[0] == None:
            st.image(results_grey_psp[0])
        st.text('MODEL_modelPSP_REAL72_grayscale.h5 ')
        # st.text(f'type {results_grey_psp[1]} {len(results_grey_psp[1])}')
        st.image(results_grey_psp[1])
