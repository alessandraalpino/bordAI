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
    enhanceBrightness,
    getTranslation,
    convert_colors,
    classify_user_intent
)


# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("API_KEY")

# Configure Generative AI
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# App title
st.title("BordAI 🎨🧵")
# Language selector
language = st.selectbox("Choose language / Escolha o idioma", ["en", "pt"])

# Initialize conversation state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "waiting_for_image" not in st.session_state:
    st.session_state.waiting_for_image = False

if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""

# Detect language change and reset the conversation if it changed
if "language" not in st.session_state:
    st.session_state.language = language
elif st.session_state.language != language:
    # Reset chat if the language was changed
    st.session_state.chat_history = []
    st.session_state.ai_response = ""
    st.session_state.waiting_for_image = False
    st.session_state.language = language

# Sidebar
st.sidebar.title(getTranslation("side_bar_title", language))
st.sidebar.markdown(getTranslation("side_bar_description", language))

# Button – Suggest threads
if st.sidebar.button(getTranslation("activate_tool_button", language)):
    st.session_state.waiting_for_image = True
    st.sidebar.success(getTranslation("activate_tool_success", language))

# Button – Reset conversation
if st.sidebar.button(getTranslation("reset_chat_button", language)):
    st.session_state.chat_history = []
    st.session_state.ai_response = ""
    st.session_state.waiting_for_image = False
    st.sidebar.info(getTranslation("reset_chat_success", language))


# Display initial message if chat history is empty
if not st.session_state.chat_history:
    st.session_state.chat_history.append(("assistant", getTranslation("initial_message", language)))

# Display full chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# User input
user_message = st.chat_input(getTranslation("chat_input_placeholder", language))

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
        - "1" → if the user is asking for help choosing embroidery thread colors based on an image, drawing, reference, or visual idea — OR if they mention uploading, sending, or sharing an image for color suggestions — OR if they ask which Anchor threads to use in a specific visual context.
        - "0" → for any other question not related to choosing colors based on a visual reference.

        Sentence: "{user_message}"

        Reply:
        """

        classification = model.generate_content(
            classification_prompt,
            generation_config={"max_output_tokens": 1}
        ).text.strip()

        st.write(classify_user_intent(user_message, model))

        if classification == "1":
            st.session_state.waiting_for_image = True
            assistant_reply = getTranslation("image_request", language)
        else:
                response_prompt = f"""
                You are an embroidery assistant. Be clear and helpful in your responses. Whenever possible, organize the explanation in short and clear bullet points. Try to conclude your reasoning in up to 350 tokens to avoid exceeding the response limit.
                Respond in the same language as the user's message.
                User's question: "{user_message}"
                """
                response = model.generate_content(
                    response_prompt,
                    #generation_config={"max_output_tokens": 800}
                )
                assistant_reply = response.text

        # Display assistant message and save it
        st.chat_message("assistant").write(assistant_reply)
        st.session_state.chat_history.append(("assistant", assistant_reply))

    else:
        assistant_reply = getTranslation("waiting_for_image_reminder", language)
        st.chat_message("assistant").write(assistant_reply)
        st.session_state.chat_history.append(("assistant", assistant_reply))

# Upload + image processing
if st.session_state.waiting_for_image:
    uploaded = st.file_uploader(getTranslation("upload_prompt", language), type=["png", "jpg", "jpeg"])

    if uploaded:
        original_image = Image.open(uploaded)
        st.image(original_image, caption=getTranslation("original_image_caption", language), use_container_width=True)

        # Preprocessing options
        st.markdown(f'### {getTranslation("adjust_image_title", language)}')
        remove_bg = st.checkbox(getTranslation("remove_bg", language))
        brightness = st.slider(getTranslation("brightness", language), 0, 100, 0, step=1)
        num_colors = st.number_input(getTranslation("num_colors", language),
                                     min_value=1, max_value=15, value=5)
        thread_brand = st.selectbox(getTranslation("thread_brand", language), ["Anchor", "DMC"], index= 0)

        # Adjust image
        processed_image = original_image
        if remove_bg:
            processed_image = remove_background(processed_image)

        if brightness > 0:
            processed_image = enhanceBrightness(processed_image, brightness)

        # Before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption=getTranslation("before_image_caption", language), use_container_width=True)
        with col2:
            st.image(processed_image, caption=getTranslation("after_image_caption", language), use_container_width=True)

        if st.button(getTranslation("analyze_button", language)):
            # Color palette analysis
            st.markdown(f'### {getTranslation("palette_title", language)}')
            colors = get_colors(processed_image, num_colors)
            plot_colors(colors)

            # Load reference thread colors
            with open('anchor_colors_w_prob.json', 'r') as f:
                anchor_colors = json.load(f)
            structured_palette = {v[thread_brand]: v for v in anchor_colors.values()}
            display_color_comparison_with_probability(colors, structured_palette, thread_brand)

            # Reset state
            st.session_state.waiting_for_image = False
            final_response = getTranslation("final_response", language)
            st.chat_message("assistant").write(final_response)
            st.session_state.chat_history.append(("assistant", final_response))
