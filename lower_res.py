from PIL import Image
import os
import cv2

def lower(path, img_name):
    img = Image.open(path)
    h = img.size[1]
    w = img.size[0]

    res = img.resize((w//2, h//2))

    res.save(f"dataset/landscape/lowered/{img_name}_lower.jpg")
    
dataset = os.listdir(r"dataset\landscape\ori")

for img_name in dataset:
    path = os.path.join("dataset\landscape\ori", img_name)
    lower(path, img_name)