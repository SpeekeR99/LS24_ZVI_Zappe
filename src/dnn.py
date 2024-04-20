import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("img/pebbles.jpg")
(H, W) = img.shape[:2]
plt.imshow(img)
plt.show()

protoPath = "hed_model/deploy.prototxt"
modelPath = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

mean_pixel_values = np.average(img, axis=(0, 1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H), mean=mean_pixel_values, swapRB=True, crop=False)

blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)
plt.imshow(blob_for_plot)
plt.show()

net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]
hed = (255 * hed).astype("uint8")

plt.imshow(hed, cmap='gray')
plt.show()
