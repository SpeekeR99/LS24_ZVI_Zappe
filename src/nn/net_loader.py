import cv2
import torch

from src.nn.my_canny import CannyEdgeDetectionNet
from src.nn.my_cnn import EdgeDetectionNet
from src.nn.my_hed import HolisticallyNestedEdgeDetectionNet


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


class NetLoader:
    @staticmethod
    def load_net(model_name):
        if model_name == "canny":
            net = CannyEdgeDetectionNet(threshold=0.8)
        elif model_name == "cnn":
            net = EdgeDetectionNet()
        elif model_name == "hed":
            net = HolisticallyNestedEdgeDetectionNet()
        elif model_name == "baseline":
            protoPath = "../../models/hed_model/deploy.prototxt"
            modelPath = "../../models/hed_model/hed_pretrained_bsds.caffemodel"
            net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
            cv2.dnn_registerLayer("Crop", CropLayer)
        else:
            raise ValueError("Model not found")

        return net

    @staticmethod
    def load_weights(model, weights_path):
        model.load_state_dict(torch.load(weights_path))
        return model
