import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple

IMAGE_SIZE = (640, 480)
INPUT_CHANNELS = 3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_PATH = "onnx/iris_seg_initial.onnx"
INPUT_IMAGE = "img/sample.png"

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def preprocess(
    image: np.ndarray, input_resolution: Tuple[int, int], nn_input_channels: int
) -> np.ndarray:
    nn_input = cv2.resize(image.astype(float), input_resolution)
    nn_input = np.divide(nn_input, 255)  

    nn_input = np.expand_dims(nn_input, axis=-1)
    nn_input = np.tile(nn_input, (1, 1, nn_input_channels))

    means = np.array([0.485, 0.456, 0.406]) if nn_input_channels == 3 else 0.5
    stds = np.array([0.229, 0.224, 0.225]) if nn_input_channels == 3 else 0.5

    nn_input -= means
    nn_input /= stds

    nn_input = nn_input.transpose(2, 0, 1)
    nn_input = np.expand_dims(nn_input, axis=0)

    return nn_input


# Load, preprocess, and prepare an infrared image for model inference
def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found.")

    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_resized = np.stack([img_resized] * 3, axis=-1)
    img_resized = img_resized.astype(np.float32) / 255.0

    img_resized = (img_resized - MEAN) / STD

    img_resized = img_resized.transpose(2, 0, 1)
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized, cv2.resize(img, IMAGE_SIZE)


# Perform inference using ONNX model and return segmentation output
def run_inference(image_tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: image_tensor})
    return outputs[0]


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
    image_tensor, ir_image = preprocess_image(INPUT_IMAGE)
    output_tensor = run_inference(image_tensor)
    print(output_tensor)
    segmented_map = np.squeeze(output_tensor, axis=0)

    plot_segmentation_map(segmented_map, ir_image)
