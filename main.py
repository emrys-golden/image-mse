import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(image_a, image_b):
    """
    Compute the Mean Squared Error of the two images
    """
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err

def compare_images(image_a, image_b, title):
    m = mse(image_a, image_b)
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f" % (m))

    fig.add_subplot(1, 2, 1)
    plt.imshow(image_a, cmap=plt.cm.gray)
    plt.axis("off")

    fig.add_subplot(1, 2, 2)
    plt.imshow(image_b, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()

original = cv2.imread("images/Bau.png")
altered = cv2.imread("images/POA.png")

original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
altered = cv2.cvtColor(altered, cv2.COLOR_BGR2GRAY)

IMAGES = ("Original", altered), ("Altered", altered)

compare_images(original, altered, "BAU vs POA")
