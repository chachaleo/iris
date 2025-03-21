import onnxruntime as ort
from pipeline import pipeline, preprocess_segmap, run_segmentation, postprocess_segmap
import numpy as np
import cv2
import json


MODEL_PATH = "../onnx/iris_seg_initial.onnx"

INPUT_RESOLUTION = (640, 480)
INPUT_CHANNELS = 3

# Image of the same eye
INPUT_IMAGE = "../img/sample.png"
INPUT_IMAGE2 = "../img/sample2.png"

# Image of a different eye
INPUT_IMAGE_OTHER = "../img/sample_other.png"

def matching_onnx(iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b):
    session = ort.InferenceSession("../onnx/hamming_distance.onnx")
    
    inputs = {
        "iris_codes_a": iris_codes_a,
        "mask_codes_a": mask_codes_a,
        "iris_codes_b": iris_codes_b,
        "mask_codes_b": mask_codes_b,
    }

    match_dist = session.run(None, inputs)
    
    return match_dist


if __name__ == "__main__":
    ir_image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

    # Segmentation : generate ONNX input
    nn_input = preprocess_segmap(ir_image, INPUT_RESOLUTION, INPUT_CHANNELS).astype(
        np.float32
    )
    segmap = run_segmentation(nn_input, MODEL_PATH)
    nn_input = np.array(nn_input).reshape([-1]).tolist()

    data = dict(input_data = [nn_input])
    json.dump(data, open("../proving/segmentation/input.json", 'w' )) 

    # Run Iris Code Pipeline
    iris_codes, mask_codes = pipeline(INPUT_IMAGE, MODEL_PATH)
    iris_codes2, mask_codes2 = pipeline(INPUT_IMAGE2, MODEL_PATH)

    # Matching : generate ONNX input
    iris_codes = np.array(iris_codes).reshape([-1]).tolist()
    mask_codes = np.array(mask_codes).reshape([-1]).tolist()
    iris_codes2 = np.array(iris_codes2).reshape([-1]).tolist()
    mask_codes2 = np.array(mask_codes2).reshape([-1]).tolist()

    data = dict(input_data = [iris_codes, mask_codes, iris_codes2, mask_codes2])
    json.dump(data, open("../proving/matching/input.json", 'w' ))

    # Compute matching distance :
    match_dist_onnx = matching_onnx(iris_codes, mask_codes, iris_codes2, mask_codes2)
    print(match_dist_onnx)