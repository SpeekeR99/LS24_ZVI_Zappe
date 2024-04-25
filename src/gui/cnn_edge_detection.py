import torch

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


def canny_net_edge_detection(img):
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    blur_img, grad_mag, grad_dir, thin_edges, thresholded, early_thresholded = canny_net(img)

    res_img = (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)

    return res_img

def cnn_big_edge_detection(img):
    pass


def cnn_my_edge_detection(img):
    pass


def hed_baseline_edge_detection(img):
    pass


def hed_big_edge_detection(img):
    pass


def hed_my_edge_detection(img):
    pass


def hed_my_smoother_edge_detection(img):
    pass
