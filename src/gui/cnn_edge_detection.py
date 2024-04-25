import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from src.nn.net_loader import NetLoader


canny_net = NetLoader.load_net("canny")

cnn_big_net = NetLoader.load_net("cnn")
cnn_big_net = NetLoader.load_weights(cnn_big_net, "../../models/edge_detection_model_only_bigger_data_meta.pth")

cnn_my_net = NetLoader.load_net("cnn")
cnn_my_net = NetLoader.load_weights(cnn_my_net, "../../models/edge_detection_model_my_data.pth")

hed_baseline_net = NetLoader.load_net("baseline")

hed_big_net = NetLoader.load_net("hed")
hed_big_net = NetLoader.load_weights(hed_big_net, "../../models/edge_detection_hed_model_meta.pth")

hed_my_net = NetLoader.load_net("hed")
hed_my_net = NetLoader.load_weights(hed_my_net, "../../models/edge_detection_hed_model_my_data.pth")

hed_my_smoother_net = NetLoader.load_net("hed")
hed_my_smoother_net = NetLoader.load_weights(hed_my_smoother_net, "../../models/edge_detection_hed_model_my_data_bigger_resize.pth")

cnn_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.GaussianBlur(3, 3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
hed_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.GaussianBlur(3, 3),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def canny_net_edge_detection(img, args):
    img = img.astype(float) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    blur_img, grad_mag, grad_dir, thin_edges, thresholded, early_thresholded = canny_net(img)

    res_img = (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)
    res_img = cv2.convertScaleAbs(res_img * 255)
    return res_img


def preprocess_cnn_img(img):
    orig_w, orig_h = img.shape[:2]
    img = Image.fromarray(img)
    img = cnn_img_transform(img)
    img = img.unsqueeze(0)
    return img, orig_w, orig_h


def postprocess_cnn_output(output, orig_w, orig_h, threshold=100):
    output = output.squeeze(0).detach().numpy()
    output = output[0, :, :]  # Convert to 2D

    output = (output - output.min()) / (output.max() - output.min()) * 255
    output = output.astype("uint8")

    output = Image.fromarray(output)
    output = output.resize((orig_h, orig_w))
    output = np.array(output)

    thresholded = np.where(output > threshold, output, 0).astype(np.uint8)
    return thresholded


def cnn_big_edge_detection(img, args):
    threshold = args[0]

    img, orig_w, orig_h = preprocess_cnn_img(img)
    output = cnn_big_net(img)
    return postprocess_cnn_output(output, orig_w, orig_h, threshold)


def cnn_my_edge_detection(img, args):
    threshold = args[0]

    img, orig_w, orig_h = preprocess_cnn_img(img)
    output = cnn_my_net(img)
    return postprocess_cnn_output(output, orig_w, orig_h, threshold)


def hed_baseline_edge_detection(img, args):
    (orig_h, orig_w) = img.shape[:2]
    mean_pixel_values = np.average(img, axis=(0, 1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(orig_w, orig_h), mean=mean_pixel_values, swapRB=True, crop=False)

    hed_baseline_net.setInput(blob)
    output = hed_baseline_net.forward()
    output = output[0, 0, :, :]
    output = (255 * output).astype("uint8")

    return output


def hed_big_edge_detection(img, args):
    pass


def hed_my_edge_detection(img, args):
    pass


def hed_my_smoother_edge_detection(img, args):
    pass
