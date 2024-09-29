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
