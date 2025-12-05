# Import Library
import cv2
import numpy as np
import math 
import time
from PIL import Image
import io
import pandas as pd

import streamlit as st

# Upload Image
def upload_image(file):
  bytes_data = file.getvalue()
  array = np.frombuffer(bytes_data, np.uint8)

  img = cv2.imdecode(array, cv2.IMREAD_COLOR)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  size = img_rgb.shape
  
  st.write(f"Resolusi Image: {size[0]} x {size[1]}. Channel: {size[2]}")

  return img_rgb, size

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

# Max Resolution
def max_resolution(file, scale = 2):
  img, img_size = upload_image(file)
  target_size = (int(img_size[0]*scale), int(img_size[1]*scale), 3) # height, width, color channel

  img_bicubic = bicubic_interpolation(img, target_size, -1.0)
  kernel = np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]
  ])

  sharpened_img = cv2.filter2D(img_bicubic.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img
  # new_path = os.path.splitext(path)[0] + "_max_res" + os.path.splitext(path)[1]
  # cv2.imwrite(new_path, new_img)

# Medium Resolution
def medium_resolution(file, scale = 2):
  img, img_size = upload_image(file)
  target_size = (int(img_size[0]*scale), int(img_size[1]*scale), 3) # height, width, color channel

  img_bicubic = bicubic_interpolation(img, target_size, -0.75)
  kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ])

  sharpened_img = cv2.filter2D(img_bicubic.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img
  # new_path = os.path.splitext(path)[0] + "_med_res" + os.path.splitext(path)[1]
  # cv2.imwrite(new_path, new_img)

# Standard Resolution
def standard_resolution(file, scale = 2):
  img, img_size = upload_image(file)
  target_size = (int(img_size[0]*scale), int(img_size[1]*scale), 3) # height, width, color channel

  img_bicubic = bicubic_interpolation(img, target_size, -0.5)
  kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ])

  sharpened_img = cv2.filter2D(img_bicubic.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img
  # new_path = os.path.splitext(path)[0] + "_std_res" + os.path.splitext(path)[1]
  # cv2.imwrite(new_path, new_img)

# Min Resolution
def min_resolution(file, scale = 2):
  img, img_size = upload_image(file)
  target_size = (int(img_size[0]*scale), int(img_size[1]*scale), 3) # height, width, color channel

  img_bilinear = bilinear_interpolation(img, target_size)
  kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ])

  sharpened_img = cv2.filter2D(img_bilinear.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img
  # new_path = os.path.splitext(path)[0] + "_min_res" + os.path.splitext(path)[1]
  # cv2.imwrite(new_path, new_img)

# Baseline Resolution
def baseline_resolution(file, scale = 2):
  img, img_size = upload_image(file)
  target_size = (int(img_size[0]*scale), int(img_size[1]*scale), 3) # height, width, color channel

  img_nearest = nearest_neighbour(img, target_size)
  kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
  ])

  sharpened_img = cv2.filter2D(img_nearest.astype(np.float32), -1, kernel)
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img
  # new_path = os.path.splitext(path)[0] + "_base_res" + os.path.splitext(path)[1]
  # cv2.imwrite(new_path, new_img)
  
def custom_resolution(file, scale=2, interp="bicubic", kernel_choice="subtle", alfa_bicubic=-0.5):
  img, img_size = upload_image(file)
  target_size = (int(img_size[0]*scale), int(img_size[1]*scale), 3) # height, width, color channel

  if interp == "nearest":
    img_interp = nearest_neighbour(img, target_size)
  elif interp == "bilinear":
    img_interp = bilinear_interpolation(img, target_size)
  elif interp == "bicubic":
    img_interp = bicubic_interpolation(img, target_size, alfa_bicubic)
  else:
    print("Invalid interpolation, using bicubic interpolation")
    img_interp = bicubic_interpolation(img, target_size, alfa_bicubic)

  if kernel_choice == "subtle":
    kernel = np.array([
      [0,-1,0],
      [-1,5,-1],
      [0,-1,0]
    ])
  elif kernel_choice == "sharp":
    kernel = np.array([
      [-1,-1,-1],
      [-1,9,-1],
      [-1,-1,-1]
    ])
  else:
    print("Invalid kernel choice, using subtle kernel")
    kernel = np.array([
      [0,-1,0],
      [-1,5,-1],
      [0,-1,0]
    ])

  sharpened_img = cv2.filter2D(img_interp.astype(np.float32), -1, kernel) # diganti jadi float32 agar stabil untuk filter di cv2
  sharpened_img = np.clip(sharpened_img, 0, 255)
  new_img = cv2.cvtColor(sharpened_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

  return new_img
  # new_path = os.path.splitext(path)[0] + "_cus_res" + os.path.splitext(path)[1]
  # cv2.imwrite(new_path, new_img)
  
# Streamlit Deploy
if 'processed' not in st.session_state:
    st.session_state['processed'] = False
    
if 'resolution' not in st.session_state:
    st.session_state['resolution'] = None
    
if 'state_custom' not in st.session_state:
    st.session_state['state_custom'] = None

if 'new_img' not in st.session_state:
    st.session_state['new_img'] = None

if 'interp' not in st.session_state:
    st.session_state['interp'] = 'bicubic'

if 'kernel_choice' not in st.session_state:
    st.session_state['kernel_choice'] = 'subtle'

if 'alfa' not in st.session_state:
    st.session_state['alfa'] = -0.5

start_time = 0
end = 0

st.header("Super Resolution Demo")
st.text("Web ini adalah peningkat resolusi citra silahkan upload citra kamu dan pilih metode yang kamu inginkan!")

st.subheader("Upload Citra")
file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

if file is not None:
  st.subheader("Pengaturan Resolusi")
  resolution = st.selectbox("Pilih Pengaturan Resolusi", options=("Pilih", "Maximum Resolution", "Medium Resolution", "Standard Resolution", "Minimal Resolution", "Base Resolution", "Custom Resolution"))
  st.session_state['resolution'] = resolution
  
  if st.button("Reset Proccess"):
    st.session_state['new_img'] = None
    st.session_state['state_custom'] = None
    st.session_state['processed'] = False
    st.stop()
    
  if resolution == "Maximum Resolution" and not st.session_state['processed']:
    st.session_state['state_custom'] = None
    start_time = time.time()
    
    with st.spinner("Citra sedang di proses (Maximum)...", show_time=True):
      st.session_state['new_img'] = max_resolution(file)
      st.session_state['processed'] = True
    
    end = time.time()
  elif resolution == "Medium Resolution" and not st.session_state['processed']:
    st.session_state['state_custom'] = None
    start_time = time.time()
    
    with st.spinner("Citra sedang di proses (Medium)...", show_time=True):
      st.session_state['new_img'] = medium_resolution(file)
      st.session_state['processed'] = True
    
    end = time.time()
  elif resolution == "Standard Resolution" and not st.session_state['processed']:
    st.session_state['state_custom'] = None
    start_time = time.time()
    
    with st.spinner("Citra sedang di proses (Standard)...", show_time=True):
      st.session_state['new_img'] = standard_resolution(file)
      st.session_state['processed'] = True
    
    end = time.time()
  elif resolution == "Minimal Resolution" and not st.session_state['processed']:
    st.session_state['state_custom'] = None
    start_time = time.time()
    
    with st.spinner("Citra sedang di proses (Minimal)...", show_time=True):
      st.session_state['new_img'] = min_resolution(file)
      st.session_state['processed'] = True
    
    end = time.time()
  elif resolution == "Base Resolution" and not st.session_state['processed']:
    st.session_state['state_custom'] = None
    start_time = time.time()
    
    with st.spinner("Citra sedang di proses (Base)...", show_time=True):
      st.session_state['new_img'] = baseline_resolution(file)
      st.session_state['processed'] = True
    
    end = time.time()
  elif resolution == "Custom Resolution" and not st.session_state['processed']:
    st.session_state['state_custom'] = True
  else:
    print("")

  if st.session_state['state_custom']:
    st.subheader("Metode Interpolation")
    choice = st.selectbox("Pilih Metode Interpolation", options=("Bicubic", "Bilinear", "Nearest Neighbour"))
    if choice == "Bicubic":
        interp = "bicubic"
    elif choice == "Bilinear":
        interp = "bilinear"
    elif choice == "Nearest Neighbour":
        interp = "nearest"
    st.session_state['interp'] = interp

    st.subheader("Kernel")
    choice = st.selectbox("Pilih Kernel", options=("Lembut", "Tajam"))
    if choice == "Lembut":
        kernel_choice = "subtle"
    elif choice == "Tajam":
        kernel_choice = "sharp"
    st.session_state['kernel_choice'] = kernel_choice

    if interp == "bicubic":
      st.text("Pilih Nilai Alpha")
      alfa = st.slider('Alpha', min_value=-1.0, max_value=0.0, value=-0.05, format = '%.2f', step=0.01)
      st.write(f"Selected Alpha: {alfa}")
      st.session_state['alfa'] = alfa
    else:
      alfa = -0.5

    if st.button("Run Custom Resolution"):
        start_time = time.time()
        with st.spinner("Citra sedang di proses (Custom)...", show_time=True):
            st.session_state['new_img'] = custom_resolution(file, interp=st.session_state['interp'], kernel_choice=st.session_state['kernel_choice'], alfa_bicubic=st.session_state['alfa'])
            st.session_state['processed'] = True
        end = time.time()

def histogram_data(array):
    intensity = np.mean(array, axis=2).flatten()
    hist, bin = np.histogram(intensity, bins=256, range=(0, 256))
    
    df = pd.DataFrame({
        'Intensitas' : bin[:-1],
        'Frekuensi': hist
    })
    
    return df

if st.session_state['new_img'] is not None:
  st.success(f"Super Resolution Berhasil dengan waktu {end-start_time:.2f} detik")

  st.text("Before Super Resolution")
  st.image(file, caption = "Before")

  st.text("After Super Resolution")
  st.image(st.session_state['new_img'], caption = "After", channels = "BGR")
  
  # Download Image
  img_dw = cv2.cvtColor(st.session_state['new_img'], cv2.COLOR_BGR2RGB)
  img_dw = Image.fromarray(img_dw)
  buffer = io.BytesIO()
  img_dw.save(buffer, format=file.type.split("/")[1].upper())
  buffer.seek(0)
  
  st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name=file.name.split('.')[0] + "_" + resolution + "." + file.type.split("/")[1],
        mime=file.type,
        icon=":material/download:",
    )
  
  st.text("Histogram Before Super Resolution")
  img_hist, _ = upload_image(file)
  hist_before = histogram_data(img_hist)
  st.bar_chart(hist_before, x='Intensitas', y='Frekuensi')
  
  st.text("Histogram After Super Resolution")
  hist_after = histogram_data(st.session_state['new_img'])
  st.bar_chart(hist_after, x='Intensitas', y='Frekuensi')