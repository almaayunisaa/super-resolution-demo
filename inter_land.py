import numpy as np
import math 
import os
import cv2

# Upload Image
def upload_image(path):
  img = cv2.imread(path)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img_rgb

# Nearest Neighbour
def nearest_neighbour(img, size):
  img_nearest = np.zeros((size[0], size[1], size[2]), dtype=np.uint8)
  scalex = size[0]/img.shape[0]
  scaley = size[1]/img.shape[1]

  for i in range(size[0]):
    for j in range(size[1]):
      y = min(math.floor(j/scalex), img.shape[1] -1)
      x = min(math.floor(i/scaley), img.shape[0]-1)
      img_nearest[i,j] = img[x, y]

  return img_nearest

# Bilinear Interpolation
def bilinear_interpolation(img, size):
  img_bilinear = np.zeros((size[0], size[1], size[2]), dtype=np.uint8)
  scalex = size[0]/img.shape[0]
  scaley = size[1]/img.shape[1]

  for i in range(size[0]):
    for j in range(size[1]):
      x = i/scalex
      y = j/scaley

      x1 = math.floor(i/scalex)
      y1 = math.floor(j/scaley)

      x2 = min(math.ceil(i/scalex), img.shape[0]-1)
      y2 = min(math.ceil(j/scaley), img.shape[1]-1)
      
      dx = x-x1
      dy = y-y1
      
      for c in range(size[2]):
        p1=img[x1, y1, c]
        p2=img[x1, y2, c]
        p3=img[x2, y1, c]
        p4=img[x2, y2, c]

        fxy1 = (1-dx)*p1+ dx*p3
        fxy2 = (1-dx)*p2+ dx*p4

        fxy = (1-dy) * fxy1 + dy * fxy2

        img_bilinear[i,j, c] = np.clip(fxy, 0, 255)

  return img_bilinear

# Bicubic Kernel
def bicubic_kernel(a, b):
  if (abs(a) >=0) & (abs(a) <=1):
        return (b+2)*(abs(a)**3)-(b+3)*(abs(a)**2)+1
  elif (abs(a) > 1) & (abs(a) <= 2):
        return b*(abs(a)**3)-(5*b)*(abs(a)**2)+(8*b)*abs(a)-4*b
  return 0

def oc(a, b, limit): # Out Of Bound Img Check
  return max(0, min(int(a+b), limit-1))

# Bicubic Interpolation
def bicubic_interpolation(img, size, alfa):
  img_bicubic = np.zeros((size[0], size[1], size[2]), dtype=np.uint8)
  hx = img.shape[1]/size[1]
  hy = img.shape[0]/size[0]

  for i in range(size[2]):
    for j in range(size[0]):
      for k in range(size[1]):
        x, y = k *hx , j*hy

        x1 = 1 + x - math.floor(x)
        x2 = x - math.floor(x)
        x3 = math.floor(x) + 1 - x
        x4 = math.floor(x) + 2 - x

        y1 = 1 + y - math.floor(y)
        y2 = y - math.floor(y)
        y3 = math.floor(y) + 1 - y
        y4 = math.floor(y) + 2 - y

        m1 = np.matrix([[bicubic_kernel(y1, alfa), bicubic_kernel(y2, alfa), bicubic_kernel(y3, alfa), bicubic_kernel(y4, alfa)]])
        m2 = np.zeros((4,4))
        for a in range(4):
          for b in range(4):
            xs = oc(math.floor(x), b-1, img.shape[1])
            ys = oc(math.floor(y), a-1, img.shape[0])
            m2[a,b] = img[ys, xs, i]
        m3 = np.matrix([[bicubic_kernel(x1, alfa), bicubic_kernel(x2, alfa), bicubic_kernel(x3, alfa), bicubic_kernel(x4, alfa)]]).T
        res =  np.dot(np.dot(m1,m2), m3)

        img_bicubic[j, k, i] = np.clip(res[0,0], 0, 255)
  return img_bicubic

dataset = os.listdir(r"dataset\landscape\lowered")

for img_name in dataset:
    path = os.path.join("dataset\landscape\lowered", img_name)
    img = upload_image(path)
    
    size = img.shape
    scale = 2
    
    target_size = (int(size[0]*scale), int(size[1]*scale), 3) # height, width, color channel
    
    img_result = nearest_neighbour(img, target_size)
    cv2.imwrite(f"dataset/landscape/interpolation_experiment/nearest/{img_name}", cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
    img_result = bilinear_interpolation(img, target_size)
    cv2.imwrite(f"dataset/landscape/interpolation_experiment/bilinear/{img_name}",cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
    img_result = bicubic_interpolation(img, target_size, -0.5)
    cv2.imwrite(f"dataset/landscape/interpolation_experiment/bicubic/-0.5/{img_name}",cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
    img_result = bicubic_interpolation(img, target_size, -0.75)
    cv2.imwrite(f"dataset/landscape/interpolation_experiment/bicubic/-0.75/{img_name}", cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))
    img_result = bicubic_interpolation(img, target_size, -1.0)
    cv2.imwrite(f"dataset/landscape/interpolation_experiment/bicubic/-1.0/{img_name}", cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR))