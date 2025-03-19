
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import run_segmentation, preprocess_segmap

INPUT_RESOLUTION = (640, 480)
INPUT_CHANNELS = 3

MODEL_PATH = "../../onnx/iris_seg_initial.onnx"
INPUT_IMAGE = "../../img/sample.png"


def plot_segmentation_map(
    segmap: np.ndarray,
    ir_image: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    fig, axs = plt.subplots(1, 4, figsize=(18, 16))

    if ir_image is not None:
        ir_image = ir_image.astype(np.float32) / 255.0
        for i in range(4):
            axs[i].imshow(ir_image, cmap="gray")

    for i in range(4):
        axs[i].imshow(segmap[i], alpha=0.5, interpolation="nearest")

    titles = ["Eyeball", "Iris", "Pupil", "Eyelashes"]
    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
        ax.axis("off")

    plt.show()

    return fig, axs


if __name__ == "__main__":

    ir_image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    image_tensor = preprocess_segmap(ir_image, INPUT_RESOLUTION, INPUT_CHANNELS).astype(np.float32)
    segmap = run_segmentation(image_tensor, MODEL_PATH)
    segmented_map = np.squeeze(segmap["output"], axis=0)

    plot_segmentation_map(segmented_map, cv2.resize(ir_image, INPUT_RESOLUTION))
