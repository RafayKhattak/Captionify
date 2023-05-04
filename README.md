# Captionify
Captionify is a web application that generates a descriptive caption for an image using an encoder-decoder architecture. The application uses a pre-trained Transformer-based vision model (ViT) as an encoder and a pre-trained language model (GPT2) as a decoder to generate highly accurate captions for uploaded images or image URLs.
## Usage
To use Captionify, simply upload an image or enter an image URL on the web interface. The tool will then use the pre-trained models to generate a descriptive caption that accurately describes the contents of the image.
## Getting Started
To install Captionify, simply clone this repository and install the necessary dependencies using pip:
```
git clone https://github.com/<username>/<repository>.git
cd <repository>
```
Install the required dependencies using the following command:
```
pip install -r requirements.txt
```
Then run the app.py file using the following command:
```
streamlit run app.py
```
This will launch the application on your local machine. You can then upload an image or enter an image URL to generate a descriptive caption.
## Architecture
Captionify uses an encoder-decoder architecture to generate captions for images. The encoder is a pre-trained Transformer-based vision model (ViT) that encodes the input image into a sequence of feature vectors. The decoder is a pre-trained language model (GPT2) that generates a descriptive caption for the image based on the encoded features.
![vit_architecture](https://user-images.githubusercontent.com/90026724/236233200-745dae6a-569f-4558-9a12-3a56b0b8a872.jpg)
## Dependencies
- streamlit
- requests
- Pillow
- transformers
- torch
## References
- This project is based on the Encoder-Decoder architecture and uses pre-trained models from the Hugging Face Transformers library.
- The application was developed using Streamlit, an open-source app framework for Machine Learning and Data Science projects.

