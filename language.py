import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
import time

# Define available language codes (MarianMT uses language pairs)
language_codes = {
    'English': 'en',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Italian': 'it',
    'Dutch': 'nl',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Hindi': 'hi'
}

def load_model(src_lang, tgt_lang):
    # Load the appropriate MarianMT model and tokenizer based on source and target languages
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate(text, model, tokenizer):
    # Tokenize and translate the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return translated_text

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Language Translator",
        page_icon="🌐",
        layout="centered",
        initial_sidebar_state="auto",
    )

    # Add custom background
    st.markdown(
        """
        <style>
        body {
            background-image: url("bg.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        .stButton > button {
            color: white;
            background-color: #4CAF50;
            border-radius: 12px;
            padding: 10px 24px;
        }
        .stTextArea {
            color:black;
            background-color: #FFFFFF;
            border: 2px solid #4CAF50;
        }
        .title {
            color: #4CAF50;
            font-family: 'Helvetica', sans-serif;
            text-align: center;
        }
        .translated-text {
            border: 2px solid #4CAF50; 
            padding: 10px; 
            border-radius: 10px; 
            color:black;
            background-color: #E8F5E9;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Title and Description
    st.markdown("<h1 class='title'>🌍 Language Translator</h1>", unsafe_allow_html=True)
    st.write("Translate text between multiple languages with ease using state-of-the-art machine translation models.")
    
    # Language selection
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("Select source language:", list(language_codes.keys()))
    with col2:
        tgt_lang = st.selectbox("Select target language:", list(language_codes.keys()))
    
    # Ensure source and target languages are different
    if src_lang == tgt_lang:
        st.error("⚠️ Source and target languages must be different.")
        return
    
    # Get input text
    text = st.text_area("Enter text to translate:", height=150, placeholder="Type text here...")
    
    if st.button("Translate"):
        if text.strip() == "":
            st.warning("⚠️ Please enter some text to translate.")
        else:
            src_code = language_codes[src_lang]
            tgt_code = language_codes[tgt_lang]
            
            try:
                # Add spinner to show loading while translating
                with st.spinner("Translating..."):
                    time.sleep(1)  # Simulating some processing time
                    # Load model and tokenizer for the selected language pair
                    model, tokenizer = load_model(src_code, tgt_code)
                    translated_text = translate(text, model, tokenizer)
                st.success(f"**Translated Text ({tgt_lang}):**")
                st.markdown(f"<div class='translated-text'>{translated_text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in translation: {str(e)}")

    # Footer image and text for additional styling
    # footer_image = Image.open('bg.jpg')  # Add any relevant image you like
    # st.image(footer_image, use_column_width=True)
    st.markdown("<p style='text-align: center;'>Powered by Transelator  🤗</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
