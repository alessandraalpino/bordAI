import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt
import cv2
import pytesseract
import re
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from functions import (
    remove_background,
    get_colors,
    plot_colors,
    display_color_comparison_with_probability,
    enhanceBrightness
)

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("API_KEY")

# Configure Generative AI
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize conversation state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "waiting_for_image" not in st.session_state:
    st.session_state.waiting_for_image = False

if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""

# App title
st.title("BordAI 🎨🧵")
# Display full chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# User input
user_message = st.chat_input("Talk to the embroidery assistant...")

if user_message:
    # Display user message
    st.chat_message("user").write(user_message)

    # Save to chat history
    st.session_state.chat_history.append(("user", user_message))

    # If we're not waiting for an image, classify the prompt
    if not st.session_state.waiting_for_image:
        classification_prompt = f"""
        You are an embroidery assistant. Your task is to classify the user's sentence.

        Reply ONLY with:
        - "1" → if the user is asking for help choosing embroidery thread colors based on an image
        - "0" → for any other type of question.

        Sentence: "{user_message}"
        """

        classification = model.generate_content(
            classification_prompt,
            generation_config={"max_output_tokens": 1}
        ).text.strip()

        if classification == "1":
            st.session_state.waiting_for_image = True
            assistant_reply = "Sure! Please upload the image you'd like to use."
        else:
            response = model.generate_content(user_message)
            assistant_reply = response.text

        # Display assistant message and save it
        st.chat_message("assistant").write(assistant_reply)
        st.session_state.chat_history.append(("assistant", assistant_reply))

    else:
        assistant_reply = "I'm waiting for the image! 😊"
        st.chat_message("assistant").write(assistant_reply)
        st.session_state.chat_history.append(("assistant", assistant_reply))

# Upload + image processing
if st.session_state.waiting_for_image:
    uploaded = st.file_uploader("Upload your image here", type=["png", "jpg", "jpeg"])

    if uploaded:
        original_image = Image.open(uploaded)
        st.image(original_image, caption="Original image", use_column_width=True)

        # Preprocessing options
        st.markdown("### Do you want to adjust the image before analysis?")
        remove_bg = st.checkbox("Remove image background")
        brightness = st.slider("Adjust brightness (%)", 0, 100, 0, step=1)
        num_colors = st.number_input("How many dominant colors do you want to extract?",
                                     min_value=1, max_value=15, value=5)

        # Adjust image
        processed_image = original_image
        if remove_bg:
            processed_image = remove_background(processed_image)

        if brightness > 0:
            processed_image = enhanceBrightness(processed_image, brightness)

        # Before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Before", use_column_width=True)
        with col2:
            st.image(processed_image, caption="After", use_column_width=True)

        if st.button("Analyze image"):
            # Color palette analysis
            st.markdown("### Detected color palette:")
            colors = get_colors(processed_image, num_colors)
            plot_colors(colors)

            # Load reference thread colors
            with open('anchor_colors_w_prob.json', 'r') as f:
                anchor_colors = json.load(f)
            structured_palette = {v["Anchor"]: v for v in anchor_colors.values()}
            display_color_comparison_with_probability(colors, structured_palette)

            # Reset state
            st.session_state.waiting_for_image = False
            response_msg = "Here are the colors I found based on your image! ✨"
            st.chat_message("assistant").write(response_msg)
            st.session_state.chat_history.append(("assistant", response_msg))
