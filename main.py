import streamlit as st
from PIL import Image
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from color_functions import (
    remove_background,
    get_colors,
    plot_colors,
    display_color_comparison_with_probability,
    enhanceBrightness,
    getTranslation

)
from chat_functions import (
    classify_user_intent,
    extract_conversion_params,
    format_color_conversion_message,
    reset_chat,
    functions
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

if "waiting_for_conversion" not in st.session_state:
    st.session_state.waiting_for_conversion = False

if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""

# Detect language change and reset the conversation if it changed
if "language" not in st.session_state:
    st.session_state.language = language
elif st.session_state.language != language:
    # Reset chat if the language was changed
    reset_chat()
    st.session_state.language = language

# Sidebar
st.sidebar.title(getTranslation("side_bar_title", language))
st.sidebar.markdown(getTranslation("side_bar_description", language))

# Button – Suggest threads
if st.sidebar.button(getTranslation("activate_tool_button", language)):
    st.session_state.waiting_for_image = True
    st.session_state.waiting_for_conversion = False
    st.sidebar.success(getTranslation("activate_tool_success", language))

# Button – Convert colors
if st.sidebar.button(getTranslation("activate_conversion_tool_button", language)):
    st.session_state.waiting_for_image = False
    st.session_state.waiting_for_conversion = True
    st.sidebar.success(getTranslation("activate_conversion_tool_success", language))
    st.session_state.chat_history.append(("assistant", getTranslation("conversion_tool_intro", language)))

# Button – Reset conversation
if st.sidebar.button(getTranslation("reset_chat_button", language)):
    reset_chat()
    st.sidebar.info(getTranslation("reset_chat_success", language))

# Display initial message if chat history is empty
if not st.session_state.chat_history:
    st.session_state.chat_history.append(("assistant", getTranslation("initial_message", language)))

# Display full chat history
for role, message in st.session_state.chat_history:
    st.chat_message(role).write(message)

# User input
user_message = st.chat_input(getTranslation("chat_input_placeholder", language))
MAX_CHARS = 500

if user_message:
    # Display user message
    st.chat_message("user").write(user_message)
    # Save to chat history
    st.session_state.chat_history.append(("user", user_message))

    if len(user_message) > MAX_CHARS:
        st.warning(getTranslation("input_too_long", language))
    elif any(keyword in user_message.lower() for keyword in ["system"]):
        warning_msg = getTranslation("prompt_injection_detected", st.session_state.language)
        st.warning(warning_msg)
        st.chat_message("assistant").write(warning_msg)
        st.session_state.chat_history.append(("assistant", warning_msg))
        reset_chat()
    else:
        if (st.session_state.waiting_for_image or st.session_state.waiting_for_conversion) and any(k in user_message.lower() for k in getTranslation("exit_keywords", language)):
            st.session_state.waiting_for_image = False
            st.session_state.waiting_for_conversion = False
            intent = "chat"
        elif st.session_state.waiting_for_image:
            intent = "image_suggestion"
        elif st.session_state.waiting_for_conversion:
            intent = "color_conversion"
        else:
            #classify intent
            intent = classify_user_intent(user_message, model, functions)

        if intent == "image_suggestion":
            st.session_state.waiting_for_image = True
            assistant_reply = getTranslation("image_request", language)

        elif intent == "color_conversion":
            #extract only params
            input_brand, output_brand, codes = extract_conversion_params(user_message, model, functions)
            assistant_reply = format_color_conversion_message(input_brand, output_brand, codes, language)

        else:
            system_prompt = f"""
            You are an embroidery assistant. Be clear and helpful in your responses. Whenever possible, organize the explanation in short and clear bullet points.
            Try to conclude your reasoning in up to 350 tokens.
            Respond in the language: {language}
            """
            full_input = f"{system_prompt}\n\nUser message:\n\"\"\"{user_message}\"\"\""

            context = [
                *[
                    {"role": role, "parts": [{"text": msg}]} for role, msg in st.session_state.chat_history
                ],
                {"role": "user", "parts": [{"text": full_input}]}
            ]

            response = model.generate_content(context,
                                            generation_config={"max_output_tokens": 2000})
            assistant_reply = response.text
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
        st.markdown(getTranslation("image_tool_intro", language))
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
            colors = get_colors(processed_image, language, num_colors)
            plot_colors(colors, language)

            # Load reference thread colors
            with open('anchor_colors_w_prob.json', 'r') as f:
                anchor_colors = json.load(f)
            structured_palette = {v[thread_brand]: v for v in anchor_colors.values()}
            display_color_comparison_with_probability(colors, structured_palette, thread_brand, language)

            # Reset state
            st.session_state.waiting_for_image = False
            final_response = getTranslation("final_response", language)
            st.chat_message("assistant").write(final_response)
            st.session_state.chat_history.append(("assistant", final_response))
