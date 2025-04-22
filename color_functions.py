import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import streamlit as st
from scipy.spatial import distance
from rembg import remove
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage import color
import json
import requests
import os

with open("translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

def get_secret(key):
    """
    Get a secret from Streamlit or fallback to .env for local development.

    This allows the app to run both on Streamlit Cloud and locally.
    """
    try:
        return st.secrets[key]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key)


def getTranslation(key, lang):
    """
    Retrieve a translation for a given key and language.
    """
    return translations.get(key, {}).get(lang, f"[{key}]")


@st.cache_data(show_spinner="🔄 Loading thread color data...")
def load_color_data(url):
    """
    Load thread color data from a URL and cache the result.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Failed to load color data: {e}")
        return {}


def convert_colors(codes, input_brand, output_brand, language):
    """
    Convert thread color codes from one brand to another.

    This function looks up color code mappings using a loaded color dataset and
    returns the corresponding codes in the target brand. If a code has no match,
    a localized "not found" message is returned.
    """

    color_data = load_color_data(get_secret("COLOR_JSON_URL"))
    if isinstance(codes, str):
        codes = [codes]

    result = {}

    for code in codes:
        matches = []

        for item in color_data.values():
            if item.get(input_brand) == code:
                output_value = item.get(output_brand)
                if output_value and output_value not in matches:
                    matches.append(output_value)

        result[code] = matches if matches else [getTranslation("not_found_text", language)]

    return result

def remove_background(image):
    """
    Remove the background from an input image.
    """
    output_image = remove(image)
    return output_image

def enhanceBrightness(image, percentage):
    """
    Increase the brightness of an image by a given percentage.
    """
    enhancer = ImageEnhance.Brightness(image)
    factor = 1 + (percentage / 100)
    bright_image = enhancer.enhance(factor)
    return bright_image

def get_colors(image, language, num_colors=5):
    """
    Extract the most dominant colors from an image using K-Means clustering.

    Converts the image to a NumPy array, handles alpha channel if present,
    and applies K-Means to identify the most important colors.
    """

    image = np.array(image)

    if len(image.shape) == 2:
        raise ValueError(getTranslation("grayscale_image_error", language))

    if image.shape[2] == 4:
        image = image[:, :, :3]

    image = image.reshape((image.shape[0] * image.shape[1], 3))

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)

    colors = np.clip(kmeans.cluster_centers_, 0, 255).astype(int)

    return colors

def plot_colors(colors, language):
    """
    Plot a horizontal bar image representing the given RGB colors.
    """
    width_per_color = 100
    height = 50
    total_width = width_per_color * len(colors)

    color_image = Image.new("RGB", (total_width, height))

    for i, color in enumerate(colors):
        for x in range(i * width_per_color, (i + 1) * width_per_color):
            for y in range(height):
                color_image.putpixel((x, y), tuple(color.astype(int)))

    st.image(
        color_image,
        caption=getTranslation("predominant_colors_caption", language),
        use_container_width=True
    )

def rgb_to_lab(rgb_color):
    """
    Convert an RGB color to CIE Lab color space.
    """
    rgb_norm = np.array(rgb_color).reshape(1, 1, 3) / 255.0
    lab_color = color.rgb2lab(rgb_norm)[0][0]

    return lab_color

def get_closest_three_colors(input_rgb_color, brand_palette):
    """
    Find the 3 closest thread colors to a given RGB color using LAB distance and probability weighting.

    Converts the input RGB color to LAB, then compares it to each color in the brand palette using Delta E
    (Euclidean distance in LAB space), adjusted by the probability of each color being a good match.
    """

    input_lab_color = rgb_to_lab(input_rgb_color)
    distances = []

    for brand_code, color_data in brand_palette.items():
        brand_rgb = color_data["RGB"]
        brand_lab = rgb_to_lab(brand_rgb)
        probability = color_data.get("Probability", 0.5)

        dist = distance.euclidean(input_lab_color, brand_lab)
        weighted_dist = dist * (1 - probability)

        distances.append((brand_code, brand_rgb, weighted_dist))

    distances.sort(key=lambda x: x[2])
    closest_colors = distances[:3]

    return [(code, rgb) for code, rgb, dist in closest_colors]

def display_color_comparison(predominant_colors, anchor_colors, thread_brand, language):
    """
    Display a visual comparison between extracted image colors and their closest thread matches.

    For each predominant RGB color, shows the original color and its 3 closest matches
    from the given thread brand, based on LAB distance and probability weighting.
    """

    num_colors = len(predominant_colors)
    fig, axs = plt.subplots(num_colors, 4, figsize=(16, 4 * num_colors))

    for i, color in enumerate(predominant_colors):
        closest_colors = get_closest_three_colors(tuple(color), anchor_colors)

        axs[i, 0].imshow([[color]])
        axs[i, 0].set_title(f'{getTranslation("predominant_color_label", language)} {i + 1}: RGB {color}')
        axs[i, 0].axis('off')

        for j, (closest_code, closest_rgb) in enumerate(closest_colors):
            axs[i, j + 1].imshow([[closest_rgb]])
            axs[i, j + 1].set_title(f'{getTranslation("code_label", language)} {thread_brand} {closest_code}')
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
