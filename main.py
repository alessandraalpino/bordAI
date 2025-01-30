import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt
# from google.colab import files
import matplotlib.pyplot as plt
import cv2
import pytesseract
import re
import json
from functions import remove_background, get_colors, plot_colors, display_color_comparison, enhanceBrightness



# Upload da imagem
uploaded = st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded:
    # Carregar a imagem original
    original_image = Image.open(uploaded)

    # Opções de processamento
    removeBackBool = st.checkbox("Remover fundo")
    percent_value = st.slider("Ajuste o brilho (%)", 0, 100, 0, step=1)

    # Verifica se há alguma modificação a ser feita
    if removeBackBool or percent_value > 0:
        image = original_image
        if removeBackBool:
            image = remove_background(image)
        if percent_value > 0:
            image = enhanceBrightness(image, percent_value)

        # Mostrar imagens lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Imagem Original", use_column_width=True)
        with col2:
            st.image(image, caption="Imagem Processada", use_column_width=True)

    else:
        # Apenas exibe a imagem original sem processamento
        st.image(original_image, caption="Imagem Original", use_column_width=True)

else:
    st.write("Por favor, envie uma imagem para visualizar.")



num_colors = st.number_input("Escolha a quantidade de cores predominantes que quer identificar",
                             value=5)

execute = st.button("Rodar")

if execute:
    colors = get_colors(image, num_colors)
    plot_colors(colors)


    with open('anchor_colors.json', 'r') as f:
        anchor_colors = json.load(f)
    #st.write(anchor_colors)

    display_color_comparison(colors, anchor_colors)  # Comparar com o dicionário anchor_colors
