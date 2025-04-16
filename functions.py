import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import streamlit as st
from scipy.spatial import distance
from rembg import remove
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
from skimage import color
import json

#Load translations file
with open("translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

def getTranslation(key, lang):
    return translations.get(key, {}).get(lang, f"[{key}]")

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




# Função para converter uma cor RGB para LAB
def rgb_to_lab(rgb_color):
    rgb_norm = np.array(rgb_color).reshape(1, 1, 3) / 255.0  # Normaliza para [0, 1]
    lab_color = color.rgb2lab(rgb_norm)[0][0]
    return lab_color

# Função para encontrar as 3 cores Anchor mais próximas no espaço LAB, considerando a probabilidade
def closest_three_anchor_colors_lab_with_probability(rgb_color, anchor_palette):
    lab_color = rgb_to_lab(rgb_color)
    distances = []

    # Calcula a métrica combinada para cada cor da paleta
    for anchor_code, color_data in anchor_palette.items():
        anchor_rgb = color_data["RGB"]
        anchor_lab = rgb_to_lab(anchor_rgb)
        probability = color_data.get("Probability", 0.5)  # Default 0.5 se não especificado

        # Calcula a distância Delta E
        dist = distance.euclidean(lab_color, anchor_lab)

        # Aplica o peso baseado na probabilidade
        weighted_dist = dist * (1 - probability)

        # Armazena o código da cor, RGB, e a métrica combinada
        distances.append((anchor_code, anchor_rgb, weighted_dist))

    # Ordena as distâncias ajustadas em ordem crescente e seleciona as 3 menores
    distances.sort(key=lambda x: x[2])  # Ordena pela métrica combinada
    closest_colors = distances[:3]  # Pega as 3 cores mais próximas

    return [(code, rgb) for code, rgb, dist in closest_colors]  # Retorna apenas o código e RGB

# Função para exibir a comparação visual entre as cores
def display_color_comparison_with_probability(predominant_colors, anchor_colors, thread_brand):
    num_colors = len(predominant_colors)

    fig, axs = plt.subplots(num_colors, 4, figsize=(16, 4 * num_colors))

    for i, color in enumerate(predominant_colors):
        # Encontrar as 3 cores Anchor mais próximas usando LAB e probabilidade
        closest_colors = closest_three_anchor_colors_lab_with_probability(tuple(color), anchor_colors)

        # Exibir a cor predominante extraída da imagem
        axs[i, 0].imshow([[color]])  # Exibe a cor
        axs[i, 0].set_title(f"Cor Predominante {i + 1}: RGB {color}")
        axs[i, 0].axis('off')

        # Exibir as 3 cores correspondentes da Anchor
        for j, (closest_code, closest_rgb) in enumerate(closest_colors):
            axs[i, j + 1].imshow([[closest_rgb]])  # Exibe a cor
            axs[i, j + 1].set_title(f"Código {thread_brand} {closest_code}")
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
