# Import required libraries
import io
import os
import streamlit as st
import requests
from PIL import Image
from model import run

# Set page configuration and title
st.set_page_config(page_title="Captionify", page_icon="🖼", layout="wide")

# Add header with title and description
st.markdown('<p style="display:inline-block;font-size:40px;font-weight:bold;">&#x1F320;Captionify </p> <p style="display:inline-block;font-size:16px;">Generate image captions with Captionify which uses a Pre-trained Transformer-based vision model (ViT) as encoder and Pre-trained language model (GPT2) as the decoder <br><br></p>', unsafe_allow_html=True)

# Get image URL input from user
img_url = st.text_input(label='Enter Image URL')

# If user has provided an image URL
if (img_url != "") and (img_url != None):
    # Open the image from the URL
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB')
    # Display the image in the app
    st.image(img, width=300)
    # Generate caption for the image using the model
    caption = run(img)
    # Display the predicted caption
    st.markdown('#### 🤖 Predicted Caption:')
    colored_text = '<span style="color:#F2EFDE;font-size:27px;">{}</span>'.format(caption)
    st.write(colored_text, unsafe_allow_html=True)


# Display option to upload an image
st.markdown('<center style="opacity: 70%">OR</center>', unsafe_allow_html=True)
img_upload = st.file_uploader(label='Upload Image', type=['jpg', 'png', 'jpeg'])

# If user has uploaded an image
if img_upload != None:
    # Read the image from the uploaded file
    img = img_upload.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    # Generate caption for the image using the model
    caption = run(img)
    # Display the image in the app
    st.image(img, width=300)
    # Display the predicted caption in colored text
    st.markdown('#### 🤖 Predicted Caption:')
    colored_text = '<span style="color:#F2EFDE;font-size:27px;">{}</span>'.format(caption)
    st.write(colored_text, unsafe_allow_html=True)

# This is a multi-line string containing CSS code that hides the Streamlit header, footer, and menu
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

# This line displays the CSS code in the Streamlit app, effectively hiding the header, footer, and menu
st.markdown(hide_st_style, unsafe_allow_html=True)

