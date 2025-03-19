import onnxruntime as ort
import numpy as np
from pipeline import pipeline

MODEL_PATH = "../onnx/iris_seg_initial.onnx"
INPUT_IMAGE = "../img/chacha.png"
INPUT_IMAGE2 = "../img/chacha2.png"
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
    iris_codes, mask_codes = pipeline(INPUT_IMAGE, MODEL_PATH)
    iris_codes2, mask_codes2 = pipeline(INPUT_IMAGE2, MODEL_PATH)
    iris_codes_other, mask_codes_other = pipeline(INPUT_IMAGE_OTHER, MODEL_PATH)

    match_dist_onnx = matching_onnx(iris_codes, mask_codes, iris_codes2, mask_codes2)
    match_dist_onnx_other = matching_onnx(iris_codes, mask_codes, iris_codes_other, mask_codes_other)

    print(match_dist_onnx)
    print(match_dist_onnx_other)