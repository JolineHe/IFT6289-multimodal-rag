import openai
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

TEXT_EMBED_MODEL = "text-embedding-3-small"
TEXT_EMBED_SIZE = 1536
IMG_EMBED_SIZE = 512
CLIP_MODEL_PATH = "openai/clip-vit-base-patch32"
os.environ['CURL_CA_BUNDLE'] = '' # for image encoder correctly being used


def get_text_embedding(text):
    if not text or not isinstance(text, str):
        print("text is not a string")
        return None
    try:
        embedding = openai.embeddings.create(
            input=text,
            model=TEXT_EMBED_MODEL, dimensions=TEXT_EMBED_SIZE).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_text_embedding: {e}")
        return None


def get_img_embedding(img_path):
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, use_fast=True)
    try:
        import re
        import requests
        
        # Check if img_path is a URL using regex
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        print(img_path)
        if url_pattern.match(img_path):
            try:
                image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
            except:
                image = Image.open("files/error.jpg").convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")
            
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)
        return image_embedding.squeeze().numpy().tolist() # in shape (512,)
    except Exception as e:
        print(f"Error in get_img_embedding: {e}")
        return None