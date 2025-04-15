from PIL import Image
img = Image.open("fatperson.jpg").convert("RGB")
img.save("fatpersonColour.jpg")
