import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
# from nilearn import datasets
# from nilearn import plotting
import torch
from transformers import CLIPModel, CLIPProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda() # type: ignore
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# NB Make sure to change the paths accordingly
embeddings = {}
for img_path in tqdm(sorted(Path("data/train_data/subj01/training_split/training_images").glob("*.png"))):
    img = Image.open(img_path)
    inputs = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()
    embeddings[img_path.stem] = emb

np.save("data/clip_embeddings/train_vitl14.npy", embeddings) # pyright: ignore[reportArgumentType]

embeddings = {}
for img_path in tqdm(sorted(Path("data/train_data/subj01/test_split").glob("*.png"))):
    img = Image.open(img_path)
    inputs = processor(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()
    embeddings[img_path.stem] = emb

np.save("data/clip_embeddings/test_vitl14.npy", embeddings) # type: ignore
