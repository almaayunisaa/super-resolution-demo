import numpy as np
import math 
import os
import cv2

# Upload Image
def upload_image(path):
  img = cv2.imread(path)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img_rgb

def kernel_lembut(img):
  kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ])

  sharpened_img = cv2.filter2D(img.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img

def kernel_tajam(img):
  kernel = np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]
  ])

  sharpened_img = cv2.filter2D(img.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img

dataset = os.listdir(r"dataset\landscape\interpolation_experiment\bicubic\-1.0")

for img_name in dataset:
    path = os.path.join("dataset", "landscape", "interpolation_experiment", "bicubic", "-1.0", img_name)
    img = upload_image(path)
    img_result = kernel_lembut(img)
    cv2.imwrite(f"dataset/landscape/kernel_experiment/lembut/bicubic -1.0/{img_name}", img_result)
    img_result = kernel_tajam(img)
    cv2.imwrite(f"dataset/landscape/kernel_experiment/tajam/bicubic -1.0/{img_name}", img_result)
    
    