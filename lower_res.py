from PIL import Image

img = Image.open("ori.jpg")
h = img.size[1]
w = img.size[0]
img.show()
print(f"Height : {h}, Width : {w}")

res = img.resize((w//2, h//2))
res.show()
print(f"Height : {res.size[1]}, Width : {res.size[0]}")

res.save("ori_lower_2times.jpg")