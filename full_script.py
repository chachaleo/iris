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
    inv_bisectors_center_slope = np.fliplr(bisectors_second_points - bisectors_first_points)
    inv_bisectors_center_slope[:, 1] = -inv_bisectors_center_slope[:, 1]
    norm = np.linalg.norm(inv_bisectors_center_slope, axis=1)
    inv_bisectors_center_slope[:, 0] /= norm
    inv_bisectors_center_slope[:, 1] /= norm
    first_bisectors_point = bisectors_center - inv_bisectors_center_slope
    second_bisectors_point = bisectors_center + inv_bisectors_center_slope

    return first_bisectors_point, second_bisectors_point

def find_best_intersection(fst_points: np.ndarray, sec_points: np.ndarray) -> Tuple[float, float]:
    norm_bisectors = (sec_points - fst_points) / np.linalg.norm(sec_points - fst_points, axis=1)[:, np.newaxis]
    projections = np.eye(norm_bisectors.shape[1]) - norm_bisectors[:, :, np.newaxis] * norm_bisectors[:, np.newaxis]
    R = projections.sum(axis=0)
    q = (projections @ fst_points[:, :, np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    intersection_x, intersection_y = p
    return intersection_x.item(), intersection_y.item()



# ----- Smoothing -----
def run_smoothing(pupil_array, iris_array, eyeball_array, pupil_x, pupil_y, iris_x, iris_y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pupil_arcs = smooth(pupil_array, (pupil_x, pupil_y))
    iris_arcs = smooth(iris_array, (iris_x, iris_y))
    return pupil_arcs, iris_arcs, eyeball_array

def smooth(polygon: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
    arcs, num_gaps = cut_into_arcs(polygon, center_xy)
    arcs = (
        smooth_circular_shape(arcs[0], center_xy)
        if num_gaps == 0
        else np.vstack([smooth_arc(arc, center_xy) for arc in arcs if len(arc) >= 2])
    )
    return arcs

def cut_into_arcs(polygon: np.ndarray, center_xy: Tuple[float, float], gap_threshold: float = 10.0) -> Tuple[List[np.ndarray], int]:
    rho, phi = cartesian2polar(polygon[:, 0], polygon[:, 1], *center_xy)
    phi, rho = sort_two_arrays(phi, rho)
    differences = np.abs(phi - np.roll(phi, -1))
    differences[-1] = 2 * np.pi - differences[-1]
    gap_indices = np.argwhere(differences > np.radians(gap_threshold)).flatten()
    if gap_indices.size < 2:
        return [polygon], gap_indices.size
    gap_indices += 1
    phi, rho = np.split(phi, gap_indices), np.split(rho, gap_indices)
    arcs = [
        np.column_stack(polar2cartesian(rho_coords, phi_coords, *center_xy))
        for rho_coords, phi_coords in zip(rho, phi)
    ]
    if len(arcs) == gap_indices.size + 1:
        arcs[0] = np.vstack([arcs[0], arcs[-1]])
        arcs = arcs[:-1]
    return arcs, gap_indices.size

def smooth_arc(vertices: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
    rho, phi = cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)
    phi, rho = sort_two_arrays(phi, rho)
    idx = find_start_index(phi)
    offset = phi[idx]
    relative_phi = (phi - offset) % (2 * np.pi)
    smoothed_relative_phi, smoothed_rho = smooth_array(relative_phi, rho)
    smoothed_phi = (smoothed_relative_phi + offset) % (2 * np.pi)
    x_smoothed, y_smoothed = polar2cartesian(smoothed_rho, smoothed_phi, *center_xy)
    return np.column_stack([x_smoothed, y_smoothed])

def smooth_circular_shape(vertices: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
    rho, phi = cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)
    padded_phi = np.concatenate([phi - 2 * np.pi, phi, phi + 2 * np.pi])
    padded_rho = np.concatenate([rho, rho, rho])
    smoothed_phi, smoothed_rho = smooth_array(padded_phi, padded_rho)
    mask = (smoothed_phi >= 0) & (smoothed_phi < 2 * np.pi)
    rho_smoothed, phi_smoothed = smoothed_rho[mask], smoothed_phi[mask]
    x_smoothed, y_smoothed = polar2cartesian(rho_smoothed, phi_smoothed, *center_xy)
    return np.column_stack([x_smoothed, y_smoothed])

def cartesian2polar(xs: np.ndarray, ys: np.ndarray, center_x: float, center_y: float) -> Tuple[np.ndarray, np.ndarray]:
    x_rel: np.ndarray = xs - center_x
    y_rel: np.ndarray = ys - center_y
    C = np.vectorize(complex)(x_rel, y_rel)
    rho = np.abs(C)
    phi = np.angle(C) % (2 * np.pi)

    return rho, phi

def polar2cartesian(
    rhos: np.ndarray, phis: np.ndarray, center_x: float, center_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    xs = center_x + rhos * np.cos(phis)
    ys = center_y + rhos * np.sin(phis)

    return xs, ys

def smooth_array(phis: np.ndarray, rhos: np.ndarray, dphi: float = 1.0, kernel_size: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    kernel_offset = max(1, int((np.radians(kernel_size) / np.radians(dphi))) // 2)
    interpolated_phi = np.arange(min(phis), max(phis), np.radians(dphi))
    interpolated_rho = np.interp(interpolated_phi, xp=phis, fp=rhos, period=2 * np.pi)
    smoothed_rho = rolling_median(interpolated_rho, kernel_offset)
    if len(interpolated_phi) - 1 >= kernel_offset * 2:
        smoothed_phi = interpolated_phi[kernel_offset : -kernel_offset]
    else:
        smoothed_phi = interpolated_phi
    return smoothed_phi, smoothed_rho

def sort_two_arrays(first_list: np.ndarray, second_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    zipped_lists = zip(first_list, second_list)
    sorted_pairs = sorted(zipped_lists)
    sorted_tuples = zip(*sorted_pairs)
    first_list, second_list = [list(sorted_tuple) for sorted_tuple in sorted_tuples]
    return np.array(first_list), np.array(second_list)

def find_start_index(phi: np.ndarray) -> int:
    if not np.all((phi - np.roll(phi, 1))[1:] >= 0):
        raise "Smoothing._find_start_index phi must be sorted ascendingly!"
    phi_tmp = np.concatenate(([phi[-1] - 2 * np.pi], phi, [phi[0] + 2 * np.pi]))
    phi_tmp_left_neighbor = np.roll(phi_tmp, 1)
    dphi = (phi_tmp - phi_tmp_left_neighbor)[1:-1]
    largest_gap_index = np.argmax(dphi)
    return int(largest_gap_index)

def rolling_median(signal: np.ndarray, kernel_offset: int) -> np.ndarray:
    if signal.ndim != 1:
        raise "Smoothing._rolling_median only works for 1d arrays."
    stacked_signals: List[np.ndarray] = []
    for i in range(-kernel_offset, kernel_offset + 1):
        stacked_signals.append(np.roll(signal, i))
    stacked_signals = np.stack(stacked_signals)
    rolling_median = np.median(stacked_signals, axis=0)
    if len(rolling_median) - 1 >= kernel_offset * 2:
        rolling_median = rolling_median[kernel_offset:-kernel_offset]
    return rolling_median

# ----- Geometry Estimation -----
def run_geometry_estimation(pupil_array, iris_array, eyeball_array, pupil_x, pupil_y, iris_x, iris_y, algorithm_switch_std_threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = iris_array[:, 0], iris_array[:, 1]
    rhos, _ = cartesian2polar(xs, ys, iris_x, iris_y)
    estimated_pupil, estimated_iris = circle_extrapolation(pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y)
    radius_std = rhos.std()
    if radius_std > algorithm_switch_std_threshold:
        _, estimated_iris = ellipse_fit(pupil_array, iris_array, eyeball_array)

    return estimated_pupil, estimated_iris

# Circle extrapolation
def circle_extrapolation(pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimated_pupil = estimate(pupil_array, (pupil_x, pupil_y))
    estimated_iris = estimate(iris_array, (iris_x, iris_y))
    return estimated_pupil, estimated_iris

def estimate(vertices: np.ndarray, center_xy: Tuple[float, float], dphi: float = 0.9) -> np.ndarray:
    rhos, phis = cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)
    padded_rhos = np.concatenate([rhos, rhos, rhos])
    padded_phis = np.concatenate([phis - 2 * np.pi, phis, phis + 2 * np.pi])
    interpolated_phis = np.arange(padded_phis.min(), padded_phis.max(), np.radians(dphi))
    interpolated_rhos = np.interp(interpolated_phis, xp=padded_phis, fp=padded_rhos, period=2 * np.pi)
    mask = (interpolated_phis >= 0) & (interpolated_phis < 2 * np.pi)
    interpolated_phis, interpolated_rhos = interpolated_phis[mask], interpolated_rhos[mask]
    xs, ys = polar2cartesian(interpolated_rhos, interpolated_phis, *center_xy)
    estimated_vertices = np.column_stack([xs, ys])
    return estimated_vertices

# Ellipse fit
def ellipse_fit(pupil_array, iris_array) ->  Tuple[np.ndarray, np.ndarray]:
    extrapolated_pupil = extrapolate(pupil_array)
    extrapolated_iris = extrapolate(iris_array)
    for point in pupil_array:
        extrapolated_pupil[find_correspondence(point, extrapolated_pupil)] = point
    return extrapolated_pupil, extrapolated_iris
    
def extrapolate(polygon_points: np.ndarray, dphi: float = 1.0) -> np.ndarray:
    (x0, y0), (a, b), theta = cv2.fitEllipse(polygon_points)
    extrapolated_polygon = parametric_ellipsis(
        a / 2, b / 2, x0, y0, np.radians(theta), round(360 / dphi)
    )
    # Rotate such that 0 degree is parallel with x-axis and array is clockwise
    roll_amount = round((-theta - 90) / dphi)
    extrapolated_polygon = np.flip(np.roll(extrapolated_polygon, roll_amount, axis=0), axis=0)
    return extrapolated_polygon

def find_correspondence(src_point: np.ndarray, dst_points: np.ndarray) -> int:
    src_x, src_y = src_point
    distance = (dst_points[:, 1] - src_y) ** 2 + (dst_points[:, 0] - src_x) ** 2
    idx = np.where(distance == distance.min())[0]
    return idx

def parametric_ellipsis(a: float, b: float, x0: float, y0: float, theta: float, nb_step: int = 100) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, nb_step)
    x_coords = x0 + b * np.cos(t) * np.sin(-theta) + a * np.sin(t) * np.cos(-theta)
    y_coords = y0 + b * np.cos(t) * np.cos(-theta) - a * np.sin(t) * np.sin(-theta)
    return np.array([x_coords, y_coords]).T

# ----- Pupil to Iris Property Estimation -----
def run_pupil_property_estimation(pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y, min_pupil_diameter: float = 1.0, min_iris_diameter: float = 150.0,) -> Tuple[float, float]:
    iris_diameter = float(np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max())
    pupil_diameter = float(np.linalg.norm(pupil_array[:, None, :] - pupil_array[None, :, :], axis=-1).max())
    center_distance = np.linalg.norm([iris_x - pupil_x, iris_y - pupil_y])
    if pupil_diameter < min_pupil_diameter:
        raise "Pupil diameter is too small!"
    if iris_diameter < min_iris_diameter:
        raise "Iris diameter is too small!"
    if pupil_diameter >= iris_diameter:
        raise "Pupil diameter is larger than/equal to Iris diameter!"
    if center_distance * 2 >= iris_diameter:
        raise "Pupil center is outside iris!"
    return pupil_diameter / iris_diameter, center_distance * 2 / iris_diameter,
    

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
    refined_pupil_array, refined_iris_array, refined_eyeball_array = run_interpolation(pupil_array, iris_array, eyeball_array)

    # Distance Filter
    pupil_array, iris_array, eyeball_array = run_distance_filter(refined_pupil_array, refined_iris_array, refined_eyeball_array, noise_mask)

    # Eye Orientation
    angle = run_eye_orientation(pupil_array, iris_array, eyeball_array)

    # Eye Center Estimation
    pupil_x, pupil_y, iris_x, iris_y = run_eye_center_estimation(pupil_array, iris_array, eyeball_array)

    # Smoothing
    pupil_arcs, iris_arcs, eyeball_array = run_smoothing(pupil_array, iris_array, eyeball_array, pupil_x, pupil_y, iris_x, iris_y)

    # Geometry Estimation
    estimated_pupil, estimated_iris = run_geometry_estimation(pupil_array, iris_array, eyeball_array, pupil_x, pupil_y, iris_x, iris_y)
    print(estimated_pupil, estimated_iris)

    # Pupil to Iris Property Estimation
    pupil_to_iris_diameter_ratio, pupil_to_iris_center_dist_ratio = run_pupil_property_estimation(pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y)
    print(pupil_to_iris_diameter_ratio, pupil_to_iris_center_dist_ratio)



    
    #with open("full_output.txt", "w") as f:
    #    np.set_printoptions(threshold=np.inf)
    #    print(output_tensor, file=f)


