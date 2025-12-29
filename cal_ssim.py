from skimage.metrics import structural_similarity as ssim
import cv2
import os
from PIL import Image
import csv

all_data = []

def calculate_ssim(imageA, imageB):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(grayA, grayB, full=True)
    return score

ori = cv2.imread("ori.jpg")
hasil_folder = os.listdir("hasil")

print("SSIM Score")
for img_name in hasil_folder:
    path = os.path.join("hasil", img_name)
    img = cv2.imread(path)
    
    score = calculate_ssim(ori, img)
    data = {
        "name" : img_name,
        "score" : score
    }
    
    all_data.append(data)
    print(f"{img_name} : {score}")

with open('ssim_score.csv', 'w', newline='') as csvfile:
    fieldnames = ['name', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_data)