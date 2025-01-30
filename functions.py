import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import streamlit as st
from scipy.spatial import distance
from rembg import remove
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt



def remove_background(image):
    output_image = remove(image)
    return output_image


def enhanceBrightness(image, percentage):
    enhancer = ImageEnhance.Brightness(image)
    factor = 1 + (percentage / 100)
    bright_image = enhancer.enhance(factor)
    return bright_image

def get_colors(image, num_colors=5):

    # Converter a imagem para um array NumPy
    image = np.array(image)

    # # Verificar a forma da imagem
    # print("Dimensões da imagem:", image.shape)

    # Verificar se a imagem tem 2 dimensões (escala de cinza)
    if len(image.shape) == 2:
        raise ValueError("A imagem está em escala de cinza. Deve ser uma imagem RGB ou RGBA.")

    # Se a imagem tem 4 canais (RGBA), remover o canal alfa (transparência)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Reshape para lista de pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)

    # Normalizar os valores para o intervalo de 0 a 255 e arredondar
    colors = np.clip(kmeans.cluster_centers_, 0, 255).astype(int)
    return colors




def plot_colors(colors):
    # Definir as dimensões da imagem (largura proporcional ao número de cores)
    width_per_color = 100
    height = 50
    total_width = width_per_color * len(colors)

    # Criar uma imagem em branco
    color_image = Image.new("RGB", (total_width, height))

    # Preencher a imagem com as cores fornecidas
    for i, color in enumerate(colors):
        for x in range(i * width_per_color, (i + 1) * width_per_color):
            for y in range(height):
                color_image.putpixel((x, y), tuple(color.astype(int)))  # Converter as cores para int

    # Exibir a imagem diretamente no Streamlit
    st.image(color_image, caption="Cores Predominantes", use_column_width=True)



# Função para encontrar a cor Anchor mais próxima
def closest_anchor_color(rgb_color, anchor_palette):
    """
    Dada uma cor em RGB, encontra a cor mais próxima na paleta da Anchor.

    Args:
    rgb_color: tupla com valores (R, G, B) da cor.
    anchor_palette: dicionário onde as chaves são códigos da Anchor e os valores são tuplas (R, G, B).

    Retorna:
    O código da cor mais próxima e a cor RGB correspondente.
    """
    closest_color = None
    min_distance = float('inf')

    # Itera sobre as cores da paleta da Anchor
    for anchor_code, anchor_rgb in anchor_palette.items():
        # Calcula a distância entre a cor da imagem e a cor da paleta
        dist = distance.euclidean(rgb_color, anchor_rgb)

        # Verifica se esta cor é a mais próxima até agora
        if dist < min_distance:
            min_distance = dist
            closest_color = anchor_code

    return closest_color, anchor_palette[closest_color]


# Função para criar uma imagem de uma única cor
def create_color_image(color, width=200, height=100):
    """Cria uma imagem de uma única cor."""
    img = Image.new("RGB", (width, height), tuple([int(c) for c in color]))
    return img

# Função para exibir a comparação visual entre as cores no Streamlit
def display_color_comparison(predominant_colors, anchor_colors):
    for i, color in enumerate(predominant_colors):
        # Encontrar a cor Anchor mais próxima (presumindo que você já tem essa função)
        closest_code, closest_rgb = closest_anchor_color(tuple(color), anchor_colors)

        # Criar imagem da cor predominante e da cor Anchor mais próxima
        predominant_img = create_color_image(color)
        anchor_img = create_color_image(closest_rgb)

        # Criar colunas para exibir lado a lado
        col1, col2 = st.columns(2)

        # Exibir a cor predominante na primeira coluna
        with col1:
            st.write(f"**Cor Predominante {i + 1}: RGB {color}**")
            st.image(predominant_img, width=200)

        # Exibir a cor Anchor correspondente na segunda coluna
        with col2:
            st.write(f"**Cor Anchor mais próxima: Código {closest_code}, RGB {closest_rgb}**")
            st.image(anchor_img, width=200)

# Exemplo de uso
# display_color_comparison(colors, anchor_colors)  # Comparar com o dicionário anchor_colors
