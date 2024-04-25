import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from src.nn.net_loader import NetLoader


# Canny edge detection model (analytical weights)
canny_net = NetLoader.load_net("canny")

# CNN edge detection model (big data, metacentrum training)
cnn_big_net = NetLoader.load_net("cnn")
# CNN edge detection model (big data, metacentrum training)
cnn_big_net = NetLoader.load_weights(cnn_big_net, "../../models/edge_detection_model_only_bigger_data_meta.pth")

# CNN edge detection model (my data, local training)
cnn_my_net = NetLoader.load_net("cnn")
# CNN edge detection model (my data, local training)
cnn_my_net = NetLoader.load_weights(cnn_my_net, "../../models/edge_detection_model_my_data.pth")

# HED "baseline" edge detection model (opencv caffe)
hed_baseline_net = NetLoader.load_net("baseline")

# HED edge detection model (big data, metacentrum training)
hed_big_net = NetLoader.load_net("hed")
# HED edge detection model (big data, metacentrum training)
hed_big_net = NetLoader.load_weights(hed_big_net, "../../models/edge_detection_hed_model_meta.pth")

# HED edge detection model (my data, local training)
hed_my_net = NetLoader.load_net("hed")
# HED edge detection model (my data, local training)
hed_my_net = NetLoader.load_weights(hed_my_net, "../../models/edge_detection_hed_model_my_data.pth")

# HED edge detection model (my data, local training, 1024x1024)
hed_my_smoother_net = NetLoader.load_net("hed")
# HED edge detection model (my data, local training, 1024x1024)
hed_my_smoother_net = NetLoader.load_weights(hed_my_smoother_net, "../../models/edge_detection_hed_model_my_data_bigger_resize.pth")

# Image transformations for CNN models
cnn_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.GaussianBlur(3, 3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Image transformations for HED models
hed_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.GaussianBlur(3, 3),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def canny_net_edge_detection(img, args):
    """
    Canny edge detection using analytical weights
    :param img: Input image
    :param args: Additional arguments (UNUSED, but required for compatibility)
    :return: Edge detection result
    """
    img = img.astype(float) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    blur_img, grad_mag, grad_dir, thin_edges, thresholded, early_thresholded = canny_net(img)

    res_img = (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)
    res_img = cv2.convertScaleAbs(res_img * 255)
    return res_img


def preprocess_img(img, transform):
    """
    Preprocess image for CNN models
    :param img: Input image
    :param transform: Image transformation
    :return: Preprocessed image, original width, original height
    """
    orig_w, orig_h = img.shape[:2]
    img = Image.fromarray(img)
    img = transform(img)
    img = img.unsqueeze(0)
    return img, orig_w, orig_h


def postprocess_output(output, orig_w, orig_h, threshold=100):
    """
    Postprocess output of CNN models
    :param output: Output from model (tensor)
    :param orig_w: Original width
    :param orig_h: Original height
    :param threshold: Threshold for edge detection
    :return: Edge detection result
    """
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
    """
    CNN edge detection using big data model
    :param img: Input image
    :param args: Additional arguments (threshold)
    :return: Edge detection result
    """
    threshold = args[0]

    img, orig_w, orig_h = preprocess_img(img, cnn_img_transform)
    output = cnn_big_net(img)
    return postprocess_output(output, orig_w, orig_h, threshold)


def cnn_my_edge_detection(img, args):
    """
    CNN edge detection using my data model
    :param img: Input image
    :param args: Additional arguments (threshold)
    :return: Edge detection result
    """
    threshold = args[0]

    img, orig_w, orig_h = preprocess_img(img, cnn_img_transform)
    output = cnn_my_net(img)
    return postprocess_output(output, orig_w, orig_h, threshold)


def hed_baseline_edge_detection(img, args):
    """
    HED edge detection using baseline model
    :param img: Input image
    :param args: Additional arguments (UNUSED, but required for compatibility)
    :return: Edge detection result
    """
    (orig_h, orig_w) = img.shape[:2]
    mean_pixel_values = np.average(img, axis=(0, 1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(orig_w, orig_h), mean=mean_pixel_values, swapRB=True, crop=False)

    hed_baseline_net.setInput(blob)
    output = hed_baseline_net.forward()
    output = output[0, 0, :, :]
    output = (255 * output).astype("uint8")

    return output


def hed_big_edge_detection(img, args):
    """
    HED edge detection using big data model
    :param img: Input image
    :param args: Additional arguments (threshold)
    :return: Edge detection result
    """
    threshold = args[0]

    img, orig_w, orig_h = preprocess_img(img, hed_img_transform)
    output = hed_big_net(img)
    return postprocess_output(output, orig_w, orig_h, threshold)


def hed_my_edge_detection(img, args):
    """
    HED edge detection using my data model
    :param img: Input image
    :param args: Additional arguments (threshold)
    :return: Edge detection result
    """
    threshold = args[0]

    img, orig_w, orig_h = preprocess_img(img, hed_img_transform)
    output = hed_my_net(img)
    return postprocess_output(output, orig_w, orig_h, threshold)


def hed_my_smoother_edge_detection(img, args):
    """
    HED edge detection using my data model with 1024x1024 input size
    :param img: Input image
    :param args: Additional arguments (threshold)
    :return: Edge detection result
    """
    threshold = args[0]

    img, orig_w, orig_h = preprocess_img(img, hed_img_transform)
    output = hed_my_smoother_net(img)
    return postprocess_output(output, orig_w, orig_h, threshold)
