import requests
from bs4 import BeautifulSoup
from random import randint
from random import random
import os
import urllib.request
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def getImage():
    # instruments = ('Violin', 'Guitar', 'Saxophone', 'Guitar', 'Frenchhorn', 'Flute_Instrument', 'Erhu', 'Bassoon')

    instruments = ('Violin', 'Guitar', 'Saxophone')
    x = str(instruments[randint(0, len(instruments) - 1)])

    url = "https://www.gettyimages.com/photos/" + x + "?phrase=" + x + "&sort=mostpopular"

    print(url)

    request = requests.get(url)

    htmlSoup = BeautifulSoup(request.text, "html.parser")

    # print(htmlSoup.prettify())

    images = htmlSoup.find_all('img', {"src": True})

    imagesFinal = []

    for image in images:
        imagesFinal.append(image['src'])

    randomImage = None
    corruptedImage = True
    while corruptedImage == True:
        randomImage = imagesFinal[randint(0, len(images) - 1)]
        try:
            requests.get(randomImage)
            corruptedImage = False
        except:
            print("Corrupted img")
            corruptedImage = True

    path = "results/" + str(randint(0, 9999999)) + ".jpg"

    img = requests.get(randomImage)
    if not img.ok: getImage()
    with open(path, "wb") as f:
        f.write(img.content)

    return path


x = getImage()
print(x)
img = image.load_img(x)
plt.imshow(img)
plt.show()

model = tf.keras.models.load_model('newModel3')

img = image.load_img(x, target_size=(100, 100))
imgArray = image.img_to_array(img)  # [[[255, 0, 250]], [], [], ]
imgArray = np.expand_dims(imgArray, axis=0)  # [[[[255, 0, 250]], [], [], ]]

result = model.predict(imgArray)

# {'bassoon': 0, 'erhu': 1, 'flute': 2, 'frenchhorn': 3, 'guitar': 4, 'saxophone': 5, 'violin': 6}
# {'guitar': 0, 'saxophone': 1, 'violin': 2}
print(result)

categories = ['guitar', 'saxophone', 'violin']

maxi = -1
for i in range(0, 3):
    if result[0][i] > maxi:
        maxi = i

print(categories[maxi])

