# 🧵 BordAI – Your AI Embroidery Assistant

Being passionate about manual arts and technology, I came up with the idea of creating my own personal assistant for embroidery!

**BordAI** 🤖 is an intelligent app that helps you bring your embroidery projects to life — whether you need to pick thread colors from an image, convert between Anchor and DMC palettes, or get advice on stitches, techniques, and materials.


[👉 Try it out!](https://bordai.streamlit.app)


---

> 📖 Read the full story on Medium:
> [How I Used AI to Simplify My Embroidery Projects](https://medium.com/@yourusername/your-article-slug) TROCAR LINK


---

## ✨ Features

- 🎨 **Image‑based thread suggestions**

  Upload an image and bordAI will detect the dominant colors, then recommend the closest matching threads from the **Anchor** or **DMC** palettes.

- 🔄 **Anchor ↔ DMC color conversion**

  Enter one or more thread codes from Anchor or DMC, and instantly convert them to their equivalents in the other brand.

- 💬 **Embroidery chat assistant**

  Ask questions about stitches, techniques, fabrics, tools, etc.

- 🧠 **Automatic intent detection**

  Powered by Google Gemini, BordAI classifies your request and routes you to the right feature — whether it’s color suggestions, conversion, or chat.

- 🌐 **Multilingual support**

  Fully bilingual—switch effortlessly between **English** and **Portuguese**.

---

## 🔧 Architecture & Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini Flash 2.0
- **Image Processing & Analysis:** rembg, Pillow, scikit‑image, KMeans (scikit‑learn), scipy
- **Data Hosting:** Firebase Hosting for private color data
- **Visualization:** Matplotlib
- **Translations:** Managed via `translations.json`

---

Created by Alessandra Alpino – data scientist & embroidery enthusiast
