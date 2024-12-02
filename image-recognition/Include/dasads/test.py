from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import keras

print(cv2.__version__)

path_to_data = "./"


def convertMnistData(image):
    # image = image / 255
    image = cv2.resize(image, (28, 28))
    # print(image)
    # image = image.reshape(1, 28, 28)

    image = image.reshape(1, 28, 28, 1)

    return image


def image_show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)


def show_more(images):
    n = len(images)

    i = 0
    for k, v in images.items():
        if len(v.shape) == 3 and v.shape[2] == 3:
            v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)

        plt.subplot(1, n, i + 1)
        i += 1

        if len(v.shape) == 3 and v.shape[2] == 3:
            plt.imshow(v)
        else:
            plt.imshow(v, "gray")

        plt.title(k)
        plt.xticks([])
        plt.yticks([])
    plt.show()


# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
# wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
img = cv2.imread('0.png', cv2.IMREAD_UNCHANGED)

# https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d

kernel = np.ones((2, 2), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=10)

ret, threshed_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# img = cv2.bitwise_not(img)
img = cv2.bitwise_not(img)

# image_show(img)


contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

threshed_img = cv2.cvtColor(threshed_img, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

print(len(contours))
images = dict()
imagesPed = []
i = 0

aasd = {}
cnt = 0

for c in contours:

    x, y, w, h = cv2.boundingRect(c)

    if i == 0:
        pass
    else:
        aasd[cnt] = (x, y)
        cnt = cnt + 1

    t = w
    if w > h:
        h = w
        a = int(h / 4)
        y = y - a

    elif w < h:
        w = h
        x = int(x - (w - t) / 2)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    image = (img[y:y + h, x:x + w])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    images[i] = image
    imagesPed.append(image)
    i += 1

# image_show(img)

del imagesPed[0]
del images[0]

show_more(images)
a = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",
     4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
model = keras.models.load_model('fashion.h5')

listaPredikcija = []
for i in imagesPed:
    print(i.shape)
    # image_show(i)

    data = convertMnistData(i)
    ret = model.predict(data, batch_size=1)

    idx = np.argmax(ret);

    listaPredikcija.append(a[idx])

    print(a[idx])
    # image_show(i)

# print(f'{np.amax(ret)} -> {np.argmax(ret)} -> {ret}')


print(len(aasd))

for i in range(len(images)):
    x, y = aasd[i][0], aasd[i][1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = cv2.putText(img, listaPredikcija[i], (x, y), font, 0.5, (255, 255, 255),
                      1, cv2.LINE_AA)

image_show(img)

cv2.destroyAllWindows()