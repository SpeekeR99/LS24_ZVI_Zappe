import numpy as np
import cv2


def defined_direction_edge_detection(img, args):
    """
    Performs edge detection on the given image using the defined direction method
    :param img: Image to perform edge detection on
    :param args: Arguments for the defined direction method
    :return: Image with edges
    """
    current_defined_direction_method = args[0]
    defined_direction_horizontal = args[1]
    defined_direction_vertical = args[2]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    horizontal_kernel_1 = np.ones((3, 3))
    horizontal_kernel_2 = np.ones((3, 3))
    vertical_kernel_1 = np.ones((3, 3))
    vertical_kernel_2 = np.ones((3, 3))

    if current_defined_direction_method == 0:  # Sobel
        horizontal_kernel_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        horizontal_kernel_2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        vertical_kernel_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        vertical_kernel_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    elif current_defined_direction_method == 1:  # Prewitt
        horizontal_kernel_1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        horizontal_kernel_2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        vertical_kernel_1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        vertical_kernel_2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    elif current_defined_direction_method == 2:  # Roberts
        horizontal_kernel_1 = np.array([[1, 0], [0, -1]])
        horizontal_kernel_2 = np.array([[0, 1], [-1, 0]])
        vertical_kernel_1 = np.array([[0, 1], [-1, 0]])
        vertical_kernel_2 = np.array([[1, 0], [0, -1]])

    horizontal_1 = cv2.filter2D(img, -1, horizontal_kernel_1)
    horizontal_2 = cv2.filter2D(img, -1, horizontal_kernel_2)
    vertical_1 = cv2.filter2D(img, -1, vertical_kernel_1)
    vertical_2 = cv2.filter2D(img, -1, vertical_kernel_2)

    if defined_direction_horizontal:
        res = np.sqrt(np.power(horizontal_1, 2) + np.power(horizontal_2, 2))
    elif defined_direction_vertical:
        res = np.sqrt(np.power(vertical_1, 2) + np.power(vertical_2, 2))
    else:
        res = np.sqrt(
            np.power(horizontal_1, 2) + np.power(horizontal_2, 2) + np.power(vertical_1, 2) + np.power(vertical_2, 2))

    res = res / np.max(res) * 255
    res = res.astype(np.uint8)
    return res


def gradient_magnitude_direction_edge_detection(img, args):
    """
    Performs edge detection on the given image using the gradient magnitude and direction method
    :param img: Image to perform edge detection on
    :param args: Arguments for the gradient magnitude and direction method
    :return: Image with edges
    """
    forward_difference = args[0]
    backward_difference = args[1]
    defined_direction_horizontal = args[2]
    defined_direction_vertical = args[3]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float32(img)

    forward_difference_x = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 1):
            forward_difference_x[i, j] = abs(img[i, j + 1] - img[i, j])
    forward_difference_y = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1]):
            forward_difference_y[i, j] = abs(img[i + 1, j] - img[i, j])
    backward_difference_x = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(1, img.shape[1]):
            backward_difference_x[i, j] = abs(img[i, j] - img[i, j - 1])
    backward_difference_y = np.zeros(img.shape, dtype=np.float32)
    for i in range(1, img.shape[0]):
        for j in range(img.shape[1]):
            backward_difference_y[i, j] = abs(img[i, j] - img[i - 1, j])
    central_difference_x = backward_difference_x + forward_difference_x / 2
    central_difference_y = backward_difference_y + forward_difference_y / 2

    forward_difference_x = forward_difference_x.astype(np.uint8)
    forward_difference_y = forward_difference_y.astype(np.uint8)
    backward_difference_x = backward_difference_x.astype(np.uint8)
    backward_difference_y = backward_difference_y.astype(np.uint8)
    central_difference_x = central_difference_x.astype(np.uint8)
    central_difference_y = central_difference_y.astype(np.uint8)

    if forward_difference:
        if defined_direction_horizontal:
            return forward_difference_x
        elif defined_direction_vertical:
            return forward_difference_y
        else:
            return forward_difference_x + forward_difference_y
    elif backward_difference:
        if defined_direction_horizontal:
            return backward_difference_x
        elif defined_direction_vertical:
            return backward_difference_y
        else:
            return backward_difference_x + backward_difference_y
    else:
        if defined_direction_horizontal:
            return central_difference_x
        elif defined_direction_vertical:
            return central_difference_y
        else:
            return central_difference_x + central_difference_y


def mask_methods_edge_detection(img, args):
    """
    Performs edge detection on the given image using the mask methods
    :param img: Image to perform edge detection on
    :param args: Arguments for the mask methods
    :return: Image with edges
    """
    mask_methods_kernel = args[0]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = cv2.filter2D(img, -1, mask_methods_kernel)
    res = res / np.max(res) * 255
    res = res.astype(np.uint8)
    return res


def laplacian_operator_edge_detection(img, args):
    """
    Performs edge detection on the given image using the Laplacian operator
    :param img: Image to perform edge detection on
    :param args: Arguments for the Laplacian operator
    :return: Image with edges
    """
    laplacian_square = args[0]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3))

    if laplacian_square:
        kernel[1, 1] = -8
    else:
        kernel[0, 0] = 0
        kernel[0, 2] = 0
        kernel[2, 0] = 0
        kernel[2, 2] = 0
        kernel[1, 1] = -4
    kernel = -1 * kernel

    res = cv2.filter2D(img, -1, kernel)
    return res


def line_detection_edge_detection(img, args):
    """
    Performs line detection on the given image using the Hough transform
    :param img: Image to perform line detection on
    :param args: Arguments for the line detection (UNUSED but necessary for same function signature)
    :return: Image with red lines on it
    """
    res = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    return res


def point_detection_edge_detection(img, args):
    """
    Performs point detection on the given image
    :param img: Image to perform point detection on
    :param args: Arguments for the point detection
    :return: Image with red points on it
    """
    point_detection_threshold = args[0]

    res = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.array([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if np.sum(img[i - 1:i + 2, j - 1:j + 2] * mask) > point_detection_threshold:
                res[i, j] = [0, 0, 255]

    return res


def canny_edge_detection(img, args):
    """
    Performs edge detection on the given image using the Canny edge detector
    :param img: Image to perform edge detection on
    :param args: Arguments for the Canny edge detector
    :return: Image with edges
    """
    canny_sigma = args[0]
    canny_lower_thresh = args[1]
    canny_upper_thresh = args[2]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    size = int(2 * (np.ceil(3 * canny_sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1),
                       np.arange(-size / 2 + 1, size / 2 + 1))
    normal = 1 / (2.0 * np.pi * canny_sigma ** 2)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * canny_sigma ** 2)) / normal
    kern_size, gauss = kernel.shape[0], np.zeros_like(img, dtype=float)

    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            gauss[i, j] = np.sum(window)

    kernel, kern_size = np.array(
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3
    gx, gy = np.zeros_like(
        gauss, dtype=float), np.zeros_like(gauss, dtype=float)

    for i in range(gauss.shape[0] - (kern_size - 1)):
        for j in range(gauss.shape[1] - (kern_size - 1)):
            window = gauss[i:i + kern_size, j:j + kern_size]
            gx[i, j], gy[i, j] = np.sum(
                window * kernel.T), np.sum(window * kernel)

    gradient = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    theta += np.pi * np.int32(theta < 0)
    non_max_suppression = np.copy(gradient)

    for j in range(1, gradient.shape[0] - 1):
        for i in range(1, gradient.shape[1] - 1):
            if (0 <= theta[j, i] < 22.5 / 180 * np.pi) or (157.5 / 180 * np.pi <= theta[j, i] < np.pi):
                if gradient[j, i] < gradient[j, i - 1] or gradient[j, i] < gradient[j, i + 1]:
                    non_max_suppression[j, i] = 0
            elif 22.5 / 180 * np.pi <= theta[j, i] < 67.5 / 180 * np.pi:
                if gradient[j, i] < gradient[j - 1, i - 1] or gradient[j, i] < gradient[j + 1, i + 1]:
                    non_max_suppression[j, i] = 0
            elif 67.5 / 180 * np.pi <= theta[j, i] < 112.5 / 180 * np.pi:
                if gradient[j, i] < gradient[j - 1, i] or gradient[j, i] < gradient[j + 1, i]:
                    non_max_suppression[j, i] = 0
            elif 112.5 / 180 * np.pi <= theta[j, i] < 157.5 / 180 * np.pi:
                if gradient[j, i] < gradient[j + 1, i - 1] or gradient[j, i] < gradient[j - 1, i + 1]:
                    non_max_suppression[j, i] = 0

    non_max_suppression = non_max_suppression / np.max(non_max_suppression) * 255
    weak, strong = np.copy(non_max_suppression), np.copy(non_max_suppression)
    weak[weak < canny_lower_thresh] = 0
    weak[weak >= canny_upper_thresh] = 0
    weak[weak != 0] = 255
    strong[strong < canny_upper_thresh] = 0
    strong[strong >= canny_upper_thresh] = 255

    res = weak + strong
    res = res.astype(np.uint8)

    return res


def canny_edge_detection_opencv(img, args):
    """
    Performs edge detection on the given image using the Canny edge detector (OpenCV implementation)
    :param img: Image to perform edge detection on
    :param args: Arguments for the Canny edge detector
    :return: Image with edges
    """
    canny_lower_thresh = args[0]
    canny_upper_thresh = args[1]

    return cv2.Canny(img, canny_lower_thresh, canny_upper_thresh)


def marr_hildreth_edge_detection(img, args):
    """
    Performs edge detection on the given image using the Marr-Hildreth edge detector
    :param img: Image to perform edge detection on
    :param args: Arguments for the Marr-Hildreth edge detector
    :return: Image with edges
    """
    marr_hildreth_sigma = args[0]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    size = int(2 * (np.ceil(3 * marr_hildreth_sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1),
                       np.arange(-size / 2 + 1, size / 2 + 1))
    normal = 1 / (2.0 * np.pi * marr_hildreth_sigma ** 2)
    kernel = ((x ** 2 + y ** 2 - (2.0 * marr_hildreth_sigma ** 2)) / marr_hildreth_sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * marr_hildreth_sigma ** 2)) / normal
    kern_size = kernel.shape[0]
    log = np.zeros_like(img, dtype=float)

    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)
    zero_crossing = np.zeros_like(log)

    for i in range(log.shape[0] - (kern_size - 1)):
        for j in range(log.shape[1] - (kern_size - 1)):
            if log[i][j] == 0:
                if (log[i][j - 1] < 0 and log[i][j + 1] > 0) or (log[i][j - 1] < 0 and log[i][j + 1] < 0) or (
                        log[i - 1][j] < 0 and log[i + 1][j] > 0) or (log[i - 1][j] > 0 and log[i + 1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j - 1] > 0) or (log[i][j + 1] > 0) or (log[i - 1][j] > 0) or (log[i + 1][j] > 0):
                    zero_crossing[i][j] = 255

    zero_crossing = zero_crossing.astype(np.uint8)
    return zero_crossing
