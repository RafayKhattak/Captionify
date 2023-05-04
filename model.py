# Import required libraries
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import numpy as np

# Load pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set parameters for model generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate captions for an image
def predict(image):
  images = []

  # Convert image to RGB if not already in RGB format
  if image.mode != "RGB":
    image = image.convert(mode="RGB")

  # Append image to images list
  images.append(image)

  # Extract features from image and convert to tensor
  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  # Generate output ids using the pre-trained model
  output_ids = model.generate(pixel_values, **gen_kwargs)

  # Decode output ids to text and format captions
  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]

  # Return generated captions
  return preds

# Function to run the image captioning model and return a formatted caption
def run(image):
  # Generate captions for image
  caption = predict(image)

  # Join captions into a single string, capitalize first letter, and add period
  caption = ' '.join([str(elem) for elem in caption])
  caption = caption.capitalize()
  caption = caption + "."

  # Return formatted caption
  return caption
