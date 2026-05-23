import numpy as np
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_images = len(os.listdir("data/train_data/subj01/training_split/training_images"))

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda() # type: ignore
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# NB Make sure to change the paths accordingly
embeddings = np.zeros((9841, 768))
for i, img_path in tqdm(enumerate(sorted(Path("data/train_data/subj01/training_split/training_images").glob("*.png")))):
    img = Image.open(img_path)
    inputs = processor(images=img, return_tensors="pt").to("cuda") # type: ignore
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()
    embeddings[i] = emb

np.save("data/clip_embeddings/train_vitl14.npy", embeddings) 

embeddings = np.zeros((159, 768))
for i, img_path in tqdm(enumerate(sorted(Path("data/train_data/subj01/test_split/test_images").glob("*.png")))):
    img = Image.open(img_path)
    inputs = processor(images=img, return_tensors="pt").to("cuda") # type: ignore
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()
    embeddings[i] = emb

np.save("data/clip_embeddings/test_vitl14.npy", embeddings) 

