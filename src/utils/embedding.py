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
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    if not os.path.exists(img_path):
            return None
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)
        return image_embedding.squeeze().numpy().tolist() # in shape (512,)
    except Exception as e:
        print(f"Error in get_img_embedding: {e}")
        return None