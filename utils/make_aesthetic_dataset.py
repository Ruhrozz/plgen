import os
import pathlib
from glob import glob
from random import shuffle

import cv2
import torch
from rich.progress import track
from transformers import CLIPModel
from transformers import CLIPProcessor

from aesthetic import Classifier
from aesthetic import image_embeddings_file

torch.set_num_threads(1)
cv2.setNumThreads(1)

aesthetic_path = "aes-B32-v0.pth"
clip_name = "openai/clip-vit-base-patch32"

clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to("cuda").eval()

aes_model = Classifier(512, 256, 1)
aes_model.load_state_dict(torch.load(aesthetic_path))
aes_model = aes_model.to("cuda")

images = glob("../anime_dataset/*.jpg")
shuffle(images)

aesthetic_paths = []
for image in track(images):
    image_embeds = image_embeddings_file(image, clipmodel, clipprocessor)
    prediction = aes_model(image_embeds)
    if prediction.item() > 0.95:
        aesthetic_paths.append(image)
        os.system(f"cp {image} ./aesthetic_dataset/")
        os.system(
            f"cp {str(pathlib.Path(image).parent / pathlib.Path(image).stem)}.txt ./dataset/",
        )
