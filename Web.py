from transformers import QwenForVL, QwenTokenizer
from PIL import Image
import torch

# Load the model and tokenizer
model = QwenForVL.from_pretrained("Qwen/Qwen2-VL")
tokenizer = QwenTokenizer.from_pretrained("Qwen/Qwen2-VL")

# Load and process the image
image = Image.open('path_to_image.jpg')
inputs = tokenizer(image, return_tensors="pt")

# Run OCR
outputs = model(**inputs)
extracted_text = tokenizer.decode(outputs.logits.argmax(-1))
print(extracted_text)

import pytesseract
from PIL import Image

# Load the image
image = Image.open('path_to_image.jpg')

# Run OCR on image (for Hindi and English)
extracted_text = pytesseract.image_to_string(image, lang='hin+eng')
print(extracted_text)

import json
text_data = {"extracted_text": extracted_text}
with open('extracted_text.json', 'w') as json_file:
    json.dump(text_data, json_file)

import streamlit as st
from PIL import Image
import pytesseract

st.title('Hindi & English OCR Web App')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

keyword = st.text_input("Enter a keyword to search in the extracted text:")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    extracted_text = pytesseract.image_to_string(image, lang='hin+eng')
    st.write("Extracted Text:")
    st.text(extracted_text)

    if keyword:
        if keyword.lower() in extracted_text.lower():
            st.write(f"Keyword '{keyword}' found!")
        else:
            st.write(f"Keyword '{keyword}' not found.")
