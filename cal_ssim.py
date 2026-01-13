from skimage.metrics import structural_similarity as ssim
import cv2
import os
from PIL import Image
import csv

all_data = []

def search_ori_img(target_dataset, target_img_name):
    if target_dataset!="landscape":
        path = os.path.join("dataset", target_dataset, "ori", (target_img_name.split('.jpg_lower')[0])+".jpg")
    else:
        path = os.path.join("dataset", target_dataset, "ori", (target_img_name.split('.jpeg_lower')[0])+".jpeg")
    return cv2.imread(path) 

def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(grayA, grayB, full=True)
    return score

folder_dataset = os.listdir("dataset")

for dataset in folder_dataset:
    path = os.path.join("dataset", dataset, "kernel_experiment")
    folder_kernel = os.listdir(path)
    for kernel in folder_kernel:
        path = os.path.join("dataset", dataset, "kernel_experiment", kernel)
        folder_interpolation = os.listdir(path)
        for interpolation in folder_interpolation:
            path = os.path.join("dataset", dataset, "kernel_experiment", kernel, interpolation)
            folder_interpolation = os.listdir(path)
            folder_img = os.listdir(path)
            for img_name in folder_img:
                path = os.path.join("dataset", dataset, "kernel_experiment", kernel, interpolation, img_name)
                img = cv2.imread(path)
                ori = search_ori_img(dataset, img_name)
                ori = cv2.resize(ori, (img.shape[1], img.shape[0]))
                print(ori.shape, img.shape)
                score = calculate_ssim(ori, img)
                data = {
                    "dataset" : dataset,
                    "kernel" : kernel,
                    "interpolation" : interpolation,
                    "name" : img_name,
                    "score" : score
                }
                
                all_data.append(data)


with open('ssim_score_final.csv', 'w', newline='') as csvfile:
    fieldnames = ['dataset', 'kernel', 'interpolation', 'alpha', 'name', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_data)