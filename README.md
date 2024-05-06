# Edge Detection in Images Using Neural Networks

This project expands upon traditional edge detection methods by incorporating machine learning and neural networks.
It implements various models for edge detection, including convolutional neural networks (CNNs) and a modified Holistically-Nested Edge Detection (HED) model.

## Introduction

This project aims to detect edges in images using both traditional methods and neural network-based approaches.
It builds upon a previous semester's work on traditional edge detection techniques by adding CNNs and exploring their effectiveness in edge detection.
(You can view the original repo on my GitHub profile)

## Implemented Methods

The following edge detection methods are implemented:

*   **Official HED Model (OpenCV + Caffe):** A pre-trained HED model implemented using OpenCV and Caffe. This serves as a baseline for comparison.
*   **Custom Convolutional Neural Network:** A custom-designed CNN architecture.
*   **Canny Model with Analytical Weights:** A non-trainable model that mimics the Canny edge detection algorithm by analytically setting the weights of convolutional layers based on Gaussian blurring and Sobel operators.
*   **Custom Smaller HED Model:** A simplified version of the HED model with fewer layers and parameters.

### Method Descriptions:

*   **HED (Holistically-Nested Edge Detection):** A deep convolutional neural network focused on edge detection, capturing information at various levels of detail. It utilizes a hierarchical architecture and fusion layers to integrate information from different convolution levels, outputting a probabilistic edge map.

*   **Custom CNN Architecture:** The custom CNN consists of:
    *   A convolutional layer with a 3x3 filter, input dimension 3 (RGB), and output dimension 8.
    *   ReLU activation and max pooling with a 2x2 filter.
    *   Another convolutional layer with a 3x3 filter, input dimension 8, and output dimension 16, followed by ReLU and max pooling.
    *   A final convolutional layer with a 3x3 filter reducing the input dimension 16 to an output dimension of 1 (grayscale edge image).

*   **Canny Model with Analytical Weights:** This model utilizes pre-calculated weights based on the Canny edge detection process. It includes convolutional layers for Gaussian blur (horizontal and vertical) and Sobel masks (horizontal and vertical). Additional convolutional layers serve as rotational difference pixel filters.

*   **Smaller HED Model:** A reduced complexity HED model consisting of only 3 simplified VGG-net sub-networks.

## Implementation Details

The entire application is written in Python. The neural network implementations are done using the `torch` library.

*   **Official HED Model:**  Leverages OpenCV's ability to load pre-trained deep neural networks. The pre-trained weights can be obtained from the official HED repository or other online sources.
*   **Custom CNN and Smaller HED Model:** Trained on the datasets described in the [Datasets](#datasets) section. The output of these networks requires thresholding to produce satisfactory edge detection results.
*   **Canny Model with Analytical Weights:** Implements Gaussian blur using weights from SciPy's Gaussian function and uses Sobel masks for edge detection.

## Datasets

The following datasets were used for training and evaluation:

*   **Custom Dataset:** Created using the previous semester's work.  10 images were manually selected and processed using traditional edge detection methods (Canny, Sobel, Gradient Magnitude).
*   **Combined BIPED and UDED Datasets:** A combination of two standard edge detection datasets: BIPED ([https://xavysp.github.io/MBIPED/](https://xavysp.github.io/MBIPED/)) and UDED ([https://github.com/xavysp/UDED](https://github.com/xavysp/UDED)).

The custom CNN and smaller HED model were trained on both the custom dataset and the combined BIPED/UDED dataset.

## Experiments and Results

The primary goal was to train the CNN models to achieve results comparable to the official HED model. The models were tested on images where traditional edge detection methods struggled, particularly images with subtle or complex edges.

*   **HED Model Comparison:** The official HED model generally performed better than traditional methods.
*   **Canny Model with Analytical Weights:** Results were similar to the traditional Canny method.
*   **Custom CNN and Smaller HED Model:** The models were able to detect edges, but required post-processing (thresholding) to produce clean results.

(Refer to the original document for visual comparisons of the results from each method on specific images).

## User Guide

1.  **Installation:**

    *   Python 3.7+ is required.
    *   Install the necessary dependencies using pip:

        ```
        pip install -r requirements.txt
        ```

2.  **Running the Application:**

    *   Navigate to the `/src/gui` directory.
    *   Run the `main.py` script:

        ```
        python main.py
        ```

3.  **GUI Usage:**

    *   The GUI provides a main menu with options for "File", "Edit", "Settings", and "Help".
    *   "File" allows loading and saving images.
    *   "Edit" contains the edge detection methods (traditional and neural network-based) and preprocessing/postprocessing options (blurring and thresholding).
    *   To apply an edge detection method, select the input image, the desired method, and adjust the parameters (if any).
