import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.signal.windows import gaussian
from PIL import Image
from matplotlib import pyplot as plt


class CannyEdgeDetectionNet(nn.Module):
    def __init__(self, threshold=10.0):
        super(CannyEdgeDetectionNet, self).__init__()

        self.threshold = threshold

        filter_size = 5
        filter = gaussian(filter_size, 1).reshape([1, 1, 1, filter_size])

        self.gaussian_horizontal = nn.Conv2d(1, 1, kernel_size=(1, filter_size), padding=(0, filter_size // 2), bias=False)
        self.gaussian_horizontal.weight.data.copy_(torch.Tensor(filter))

        filter = gaussian(filter_size, 1).reshape([1, 1, filter_size, 1])

        self.gaussian_vertical = nn.Conv2d(1, 1, kernel_size=(filter_size, 1), padding=(filter_size // 2, 0), bias=False)
        self.gaussian_vertical.weight.data.copy_(torch.Tensor(filter))

        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        self.sobel_horizontal = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_horizontal.weight.data.copy_(torch.Tensor(sobel))
        self.sobel_vertical = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_vertical.weight.data.copy_(torch.Tensor(sobel.T))

        filter_0 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
        filter_45 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        filter_90 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
        filter_135 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]])
        filter_180 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
        filter_225 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
        filter_270 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
        filter_315 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional = nn.Conv2d(1, 8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional.weight.data.copy_(torch.Tensor(filters[:, None, ...]))

    def forward(self, x):
        red = x[:, 0, :, :].unsqueeze(1)
        green = x[:, 1, :, :].unsqueeze(1)
        blue = x[:, 2, :, :].unsqueeze(1)

        blur_red = self.gaussian_vertical(self.gaussian_horizontal(red))
        blur_green = self.gaussian_vertical(self.gaussian_horizontal(green))
        blur_blue = self.gaussian_vertical(self.gaussian_horizontal(blue))

        blur_img = torch.stack([blur_red, blur_green, blur_blue], dim=1)

        grad_x_red = self.sobel_horizontal(red)
        grad_x_green = self.sobel_horizontal(green)
        grad_x_blue = self.sobel_horizontal(blue)
        grad_y_red = self.sobel_vertical(red)
        grad_y_green = self.sobel_vertical(green)
        grad_y_blue = self.sobel_vertical(blue)

        grad_mag = torch.sqrt(grad_x_red ** 2 + grad_y_red ** 2 + grad_x_green ** 2 + grad_y_green ** 2 + grad_x_blue ** 2 + grad_y_blue ** 2)
        grad_dir = torch.atan2(grad_y_red + grad_y_green + grad_y_blue, grad_x_red + grad_x_green + grad_x_blue) * 180 / np.pi
        grad_dir += 180
        grad_dir = torch.round(grad_dir / 45) * 45

        all_filtered = self.directional(grad_mag)

        indices_pos = (grad_dir / 45).long() % 8
        indices_neg = ((grad_dir / 45).long() + 4) % 8

        batch_size = indices_pos.size(0)
        height = indices_pos.size(2)
        width = indices_pos.size(3)

        pixel_cnt = batch_size * height * width
        pixel_range = torch.arange(pixel_cnt).to(x.device)

        indices = (indices_pos.view(-1).data * pixel_cnt + pixel_range).long().squeeze()
        channel_pos = all_filtered.view(-1)[indices].view(batch_size, 1, height, width)

        indices = (indices_neg.view(-1).data * pixel_cnt + pixel_range).long().squeeze()
        channel_neg = all_filtered.view(-1)[indices].view(batch_size, 1, height, width)

        channel = torch.stack([channel_pos, channel_neg], dim=1)

        is_max = channel == channel.max(dim=1, keepdim=True)[0]
        is_max = is_max[:, 0, :, :]

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0

        thresholded = thin_edges.clone()
        thresholded[thin_edges < self.threshold] = 0

        early_thresholded = grad_mag.clone()
        early_thresholded[grad_mag < self.threshold] = 0

        assert grad_mag.size() == grad_dir.size() == thin_edges.size() == thresholded.size() == early_thresholded.size()

        return blur_img, grad_mag, grad_dir, thin_edges, thresholded, early_thresholded


net = CannyEdgeDetectionNet(threshold=0.8)
print("Model initialized")

torch.save(net.state_dict(), "../../models/canny_model.pth")
print("Model saved")
# net.load_state_dict(torch.load("../../models/edge_detection_model_only_bigger_data_meta.pth"))
# print("Model loaded")

raw_img = cv2.imread("../../data/img/pebbles.jpg") / 255.0
img = torch.from_numpy(raw_img).permute(2, 0, 1).unsqueeze(0).float()

blur_img, grad_mag, grad_dir, thin_edges, thresholded, early_thresholded = net(img)

res_img = (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)
plt.imshow(res_img, cmap="gray")
plt.show()
