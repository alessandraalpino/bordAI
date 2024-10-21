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
from functions import get_colors, plot_colors, display_color_comparison





# Carregar o arquivo da imagem
uploaded = st.file_uploader("Choose a file") #, type=["png", "jpg", "jpeg"])

# Verificar se um arquivo foi carregado
if uploaded is not None:
    # Carregando e mostrando a imagem com Streamlit
    image = Image.open(uploaded)
    st.image(image, use_column_width=True)
else:
    st.write("Por favor, envie uma imagem para visualizar.")

num_colors = st.number_input("Escolha a quantidade de cores predominantes que quer identificar",
                             value=5)

colors = get_colors(image, num_colors)
plot_colors(colors)


with open('anchor_colors.json', 'r') as f:
    anchor_colors = json.load(f)

display_color_comparison(colors, anchor_colors)  # Comparar com o dicionário anchor_colors
#st.write(anchor_colors)
