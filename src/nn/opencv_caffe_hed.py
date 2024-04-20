import numpy as np
import cv2
from matplotlib import pyplot as plt


class CropLayer(object):
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


img = cv2.imread("../../data/img/pebbles.jpg")
(H, W) = img.shape[:2]
plt.imshow(img)
plt.show()

protoPath = "../../models/hed_model/deploy.prototxt"
modelPath = "../../models/hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

cv2.dnn_registerLayer("Crop", CropLayer)

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
