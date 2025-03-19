import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import *

INPUT_RESOLUTION = (640, 480)
INPUT_CHANNELS = 3

MODEL_PATH = "../../onnx/iris_seg_initial.onnx"
INPUT_IMAGE = "../../img/sample.png"

def plot_normalized_iris(
    normalized_image,
    normalized_mask,
    plot_mask: bool = True,
    stretch_hist: bool = True,
    exposure_factor: float = 1.0,
):
    fig, axis = plt.subplots(*(1, 1))

    if isinstance(axis, np.ndarray):
        for individual_ax in axis.flatten():
            individual_ax.set_xticks([])
            individual_ax.set_yticks([])
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    axis.imshow(np.minimum(normalized_image * exposure_factor, 255), cmap="gray")
    if stretch_hist:
        norm = normalized_image * normalized_mask
        norm = norm.astype(np.float32)
        norm[norm == 0] = np.nan
        axis.imshow(norm, cmap="gray")

    if plot_mask:
        nm = normalized_mask.astype(np.float64)
        nm[nm == 1] = np.nan
        axis.imshow(nm, alpha=0.3, cmap="Reds", vmin=-1, vmax=3)

    plt.show()



if __name__ == "__main__":
    ir_image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

    # Segmentation
    nn_input = preprocess_segmap(ir_image, INPUT_RESOLUTION, INPUT_CHANNELS).astype(
        np.float32
    )
    segmap = run_segmentation(nn_input, MODEL_PATH)
    segmap = postprocess_segmap(
        segmap["output"],
        original_image_resolution=(ir_image.shape[1], ir_image.shape[0]),
    )

    # Segmentation Binarization
    eyeball_mask, iris_mask, pupil_mask, noise_mask = run_binarization(segmap)

    # Vectorization
    pupil_array, iris_array, eyeball_array = run_vectorization(
        eyeball_mask, iris_mask, pupil_mask
    )

    # Distance Filter
    pupil_array, iris_array, eyeball_array = run_distance_filter(
        pupil_array, iris_array, eyeball_array, noise_mask
    )

    # Eye Orientation
    angle = run_eye_orientation(eyeball_array)

    # Eye Center Estimation
    pupil_x, pupil_y, iris_x, iris_y = run_eye_center_estimation(
        pupil_array, iris_array
    )

    # Geometry Estimation
    pupil_array, iris_array = run_geometry_estimation(
        pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y
    )

    # Linear Normalization
    normalized_image, normalized_mask = run_linear_normalization(
        ir_image, noise_mask, pupil_array, iris_array, eyeball_array, angle
    )

    plot_normalized_iris(normalized_image, normalized_mask)
