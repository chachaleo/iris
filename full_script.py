import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from pydantic import NonNegativeFloat
import math

INPUT_RESOLUTION = (640, 480) 
INPUT_CHANNELS = 3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_PATH = "onnx/iris_seg_initial.onnx"
INPUT_IMAGE = "img/sample.png"  

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ----- SEGMENTATION -----
def preprocess(image: np.ndarray, input_resolution: Tuple[int, int], nn_input_channels: int) -> np.ndarray:
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

def run_segmentation(image_tensor: np.ndarray) -> dict:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: image_tensor})

    return dict(zip(output_names, outputs))

def postprocess_segmap(
    segmap: np.ndarray,
    original_image_resolution: Tuple[int, int],
) -> np.ndarray:
    segmap = np.squeeze(segmap, axis=0)
    segmap = np.transpose(segmap, (1, 2, 0))
    segmap = cv2.resize(segmap, original_image_resolution, interpolation=cv2.INTER_NEAREST)
    return segmap



# ----- BINARIZATION -----
def run_binarization(prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eyeball_preds = prediction[..., 0]
        iris_preds = prediction[..., 1]
        pupil_preds = prediction[..., 2]
        eyelashes_preds = prediction[..., 3]

        eyeball_mask = eyeball_preds >= 0.5
        iris_mask = iris_preds >= 0.5
        pupil_mask = pupil_preds >= 0.5
        eyelashes_mask = eyelashes_preds >= 0.5

        return eyeball_mask, iris_mask, pupil_mask, eyelashes_mask

# ----- VECTORIZATION -----
def run_vectorization(eyeball_mask: np.ndarray, iris_mask: np.ndarray, pupil_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eyeball_array = find_class_contours(eyeball_mask.astype(np.uint8))
    iris_array = find_class_contours(iris_mask.astype(np.uint8))
    pupil_array = find_class_contours(pupil_mask.astype(np.uint8))
    return pupil_array.astype(np.float32), iris_array.astype(np.float32), eyeball_array.astype(np.float32) 

def find_class_contours(binary_mask: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise "_find_class_contours: No contour hierarchy found at all."  
    parent_indices = np.flatnonzero(hierarchy[..., 3] == -1)
    contours = [np.squeeze(contours[i]) for i in parent_indices]

    # Applies filters
    contours = filter_polygon_areas(contours)
    if len(contours) != 1:
        raise "_find_class_contours: Number of contours must be equal to 1."
    
    return contours[0]

def filter_polygon_areas(
    polygons: List[np.ndarray], rel_tr: NonNegativeFloat = 0.03, abs_tr: NonNegativeFloat = 0.0
) -> List[np.ndarray]:
    areas = [area(polygon) if len(polygon) > 2 else 1.0 for polygon in polygons]
    area_factors = np.array(areas) / np.max(areas)
    filtered_polygons = [
        polygon
        for area, area_factor, polygon in zip(areas, area_factors, polygons)
        if area > abs_tr and area_factor > rel_tr
    ]
    return filtered_polygons

def area(array: np.ndarray, signed: bool = False) -> float:
    if len(array.shape) != 2 or array.shape[1] != 2:
        raise ValueError(f"Unable to determine the area of a polygon with shape {array.shape}. Expecting (_, 2).")
    xs, ys = array.T
    area = 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    if not signed:
        area = abs(area)
    return float(area)





# ----- Specular Reflection Detection -----

def run_specular_reflection_detection(ir_image: np.ndarray, reflection_threshold: int = 254) -> np.ndarray:
    _, reflection_segmap = cv2.threshold(
        ir_image, reflection_threshold, 255, cv2.THRESH_BINARY
    )
    reflection_segmap = (reflection_segmap / 255.0).astype(bool)
    return reflection_segmap

# ----- Interpolation -----


def run_interpolation(pupil_array, iris_array, eyeball_array, max_distance_between_boundary_points: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris_diameter = float(np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max())

    max_boundary_dist_in_px = max_distance_between_boundary_points * iris_diameter
    refined_pupil_array = interpolate_polygon_points(pupil_array.astype(np.float32), max_boundary_dist_in_px)
    refined_iris_array = interpolate_polygon_points(iris_array.astype(np.float32), max_boundary_dist_in_px)
    refined_eyeball_array = interpolate_polygon_points(eyeball_array.astype(np.float32), max_boundary_dist_in_px)


    return refined_pupil_array, refined_iris_array, refined_eyeball_array
    

def interpolate_polygon_points(polygon: np.ndarray, max_distance_between_points_px: float) -> np.ndarray:
    previous_boundary = np.roll(polygon, shift=1, axis=0)
    distances = np.linalg.norm(polygon - previous_boundary, axis=1)
    num_points = np.ceil(distances / max_distance_between_points_px).astype(int)
    x: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for (x1, y1), (x2, y2), num_point in zip(previous_boundary, polygon, num_points):
        x.append(np.linspace(x1, x2, num=num_point, endpoint=False))
        y.append(np.linspace(y1, y2, num=num_point, endpoint=False))
    new_boundary = np.stack([np.concatenate(x), np.concatenate(y)], axis=1)
    _, indices = np.unique(new_boundary, axis=0, return_index=True)
    new_boundary = new_boundary[np.sort(indices)]
    return new_boundary


# ----- Distance Filter -----

def run_distance_filter(pupil_array, iris_array, eyeball_array, noise_mask, min_distance_to_noise_and_eyeball: float = 0.005) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris_diameter = float(np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max())

    noise_and_eyeball_polygon_points_mask = noise_mask.copy()
    for eyeball_point in np.round(eyeball_array).astype(int):
        x, y = eyeball_point
        noise_and_eyeball_polygon_points_mask[y, x] = True
    min_dist_to_noise_and_eyeball_in_px = round(
        min_distance_to_noise_and_eyeball * iris_diameter
    )
    forbidden_touch_map = cv2.blur(
        noise_and_eyeball_polygon_points_mask.astype(float),
        ksize=(
            2 * min_dist_to_noise_and_eyeball_in_px + 1,
            2 * min_dist_to_noise_and_eyeball_in_px + 1,
        ),
    )
    forbidden_touch_map = forbidden_touch_map.astype(bool)
    return filter_polygon_points(forbidden_touch_map, pupil_array), filter_polygon_points(forbidden_touch_map, iris_array), eyeball_array

def filter_polygon_points(forbidden_touch_map: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
    valid_points = [not forbidden_touch_map[y, x] for x, y in np.round(polygon_points).astype(int)]
    if not any(valid_points):
        raise "No valid points after filtering polygon points!"
    return polygon_points[valid_points]



# ----- Eye Orientation -----

def run_eye_orientation(pupil_array, iris_array, eyeball_array, eccentricity_threshold: float = 0.1) -> float:
    moments = cv2.moments(eyeball_array)
    ecc = eccentricity(moments)
    if ecc < eccentricity_threshold:
        raise "The eyeball is too circular to reliably determine its orientation. "

    angle = orientation(moments)
    return angle

def orientation(moments: Dict[str, float]) -> float:
    if (moments["mu20"] - moments["mu02"]) == 0:
        if moments["mu11"] == 0:
            orientation = 0.0
        else:
            orientation = math.copysign(np.pi / 4, moments["mu11"])
    else:
        orientation = 0.5 * np.arctan(2 * moments["mu11"] / (moments["mu20"] - moments["mu02"]))
        if (moments["mu20"] - moments["mu02"]) < 0:
            orientation += np.pi / 2

        orientation = np.mod(orientation + np.pi / 2, np.pi) - np.pi / 2

    return orientation

def eccentricity(moments: Dict[str, float]) -> float:
    if moments["mu20"] + moments["mu02"] == 0:
        return 1.0
    eccentricity = ((moments["mu20"] - moments["mu02"]) ** 2 + 4 * moments["mu11"] ** 2) / (moments["mu20"] + moments["mu02"]) ** 2

    return eccentricity


# ----- Eye Center Estimation -----




def run_eye_center_estimation(pupil_array, iris_array, eyeball_array) -> Tuple[float, float, float, float]:

    pupil_diameter = float(np.linalg.norm(pupil_array[:, None, :] - pupil_array[None, :, :], axis=-1).max())
    iris_diameter = float(np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max())

    pupil_center_x, pupil_center_y = find_center_coords(pupil_array, pupil_diameter)
    iris_center_x, iris_center_y = find_center_coords(iris_array, iris_diameter)

    return pupil_center_x, pupil_center_y, iris_center_x, iris_center_y


def find_center_coords(polygon: np.ndarray, diameter: float, min_distance_between_sector_points: float = 0.75) -> Tuple[float, float]:
    min_distance_between_sector_points_in_px = min_distance_between_sector_points * diameter
    first_bisectors_point, second_bisectors_point = calculate_perpendicular_bisectors(
        polygon, min_distance_between_sector_points_in_px
    )
    return find_best_intersection(first_bisectors_point, second_bisectors_point)

def calculate_perpendicular_bisectors(
    polygon: np.ndarray, min_distance_between_sector_points_in_px: float,  num_bisectors: int = 100, max_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(142857)
    bisectors_first_points = np.empty([0, 2])
    bisectors_second_points = np.empty([0, 2])
    for _ in range(max_iterations):
        random_indices = np.random.choice(len(polygon), size=(num_bisectors, 2))
        first_drawn_points = polygon[random_indices[:, 0]]
        second_drawn_points = polygon[random_indices[:, 1]]
        norms = np.linalg.norm(first_drawn_points - second_drawn_points, axis=1)
        mask = norms > min_distance_between_sector_points_in_px
        bisectors_first_points = np.vstack([bisectors_first_points, first_drawn_points[mask]])
        bisectors_second_points = np.vstack([bisectors_second_points, second_drawn_points[mask]])
        if len(bisectors_first_points) >= num_bisectors:
            break
    else:
        "Not able to find enough random pairs of points on the arc with a large enough distance!"
        
    bisectors_first_points = bisectors_first_points[: num_bisectors]
    bisectors_second_points = bisectors_second_points[: num_bisectors]
    bisectors_center = (bisectors_first_points + bisectors_second_points) / 2
    # Flip xs with ys and flip sign of on of them to create a 90deg rotation
    inv_bisectors_center_slope = np.fliplr(bisectors_second_points - bisectors_first_points)
    inv_bisectors_center_slope[:, 1] = -inv_bisectors_center_slope[:, 1]
    # Add perpendicular vector to center and normalize
    norm = np.linalg.norm(inv_bisectors_center_slope, axis=1)
    inv_bisectors_center_slope[:, 0] /= norm
    inv_bisectors_center_slope[:, 1] /= norm
    first_bisectors_point = bisectors_center - inv_bisectors_center_slope
    second_bisectors_point = bisectors_center + inv_bisectors_center_slope
    return first_bisectors_point, second_bisectors_point

def find_best_intersection(fst_points: np.ndarray, sec_points: np.ndarray) -> Tuple[float, float]:
    norm_bisectors = (sec_points - fst_points) / np.linalg.norm(sec_points - fst_points, axis=1)[:, np.newaxis]
    # Generate the array of all projectors I - n*n.T
    projections = np.eye(norm_bisectors.shape[1]) - norm_bisectors[:, :, np.newaxis] * norm_bisectors[:, np.newaxis]
    # Generate R matrix and q vector
    R = projections.sum(axis=0)
    q = (projections @ fst_points[:, :, np.newaxis]).sum(axis=0)
    # Solve the least squares problem for the intersection point p: Rp = q
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    intersection_x, intersection_y = p
    return intersection_x.item(), intersection_y.item()





if __name__ == "__main__":

    # LOAD IR image
    ir_image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE) 

    # Segmentation
    nn_input = preprocess(ir_image, INPUT_RESOLUTION, INPUT_CHANNELS).astype(np.float32)
    segmap = run_segmentation(nn_input)
    segmap = postprocess_segmap(segmap['output'], (1440, 1080))

    # Binarization
    eyeball_mask, iris_mask, pupil_mask, noise_mask = run_binarization(segmap)

    # Vectorization
    pupil_array, iris_array, eyeball_array = run_vectorization(eyeball_mask, iris_mask, pupil_mask)
    
    # Specular Reflection Detection
    reflection_segmap = run_specular_reflection_detection(ir_image)

    # Interpolation
    # iris_diameter = float(np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max())
    refined_pupil_array, refined_iris_array, refined_eyeball_array = run_interpolation(pupil_array, iris_array, eyeball_array)

    # Distance Filter
    pupil_array, iris_array, eyeball_array = run_distance_filter(refined_pupil_array, refined_iris_array, refined_eyeball_array, noise_mask)

    # Eye Orientation
    angle = run_eye_orientation(pupil_array, iris_array, eyeball_array)
    print(angle)

    # Eye Center Estimation

    pupil_x, pupil_yy, iris_x, iris_y = run_eye_center_estimation(pupil_array, iris_array, eyeball_array)
    print( pupil_x, pupil_yy, iris_x, iris_y)
   
   
   
   #with open("full_output.txt", "w") as f:
    #    np.set_printoptions(threshold=np.inf)
    #    print(output_tensor, file=f)


