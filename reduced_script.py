import onnxruntime as ort
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from pydantic import NonNegativeFloat, PositiveInt
import math

import matching

INPUT_RESOLUTION = (640, 480)
INPUT_CHANNELS = 3

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_PATH = "onnx/iris_seg_initial.onnx"
INPUT_IMAGE = "img/chacha.png"
INPUT_IMAGE2 = "img/chacha2.png"
INPUT_IMAGE_OTHER = "img/sample_other.png"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


# ----- SEGMENTATION -----
def preprocess_segmap(
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
    segmap = cv2.resize(
        segmap, original_image_resolution, interpolation=cv2.INTER_NEAREST
    )
    return segmap


# ----- BINARIZATION -----
def run_binarization(
    prediction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
def run_vectorization(
    eyeball_mask: np.ndarray, iris_mask: np.ndarray, pupil_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eyeball_array = find_class_contours(eyeball_mask.astype(np.uint8))
    iris_array = find_class_contours(iris_mask.astype(np.uint8))
    pupil_array = find_class_contours(pupil_mask.astype(np.uint8))
    return (
        pupil_array.astype(np.float32),
        iris_array.astype(np.float32),
        eyeball_array.astype(np.float32),
    )


def find_class_contours(binary_mask: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
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
    polygons: List[np.ndarray],
    rel_tr: NonNegativeFloat = 0.03,
    abs_tr: NonNegativeFloat = 0.0,
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
        raise ValueError(
            f"Unable to determine the area of a polygon with shape {array.shape}. Expecting (_, 2)."
        )
    xs, ys = array.T
    area = 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    if not signed:
        area = abs(area)
    return float(area)


# ----- Distance Filter -----
def run_distance_filter(
    pupil_array,
    iris_array,
    eyeball_array,
    noise_mask,
    min_distance_to_noise_and_eyeball: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris_diameter = float(
        np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max()
    )

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
    return (
        filter_polygon_points(forbidden_touch_map, pupil_array),
        filter_polygon_points(forbidden_touch_map, iris_array),
        eyeball_array,
    )


def filter_polygon_points(
    forbidden_touch_map: np.ndarray, polygon_points: np.ndarray
) -> np.ndarray:
    valid_points = [
        not forbidden_touch_map[y, x] for x, y in np.round(polygon_points).astype(int)
    ]
    if not any(valid_points):
        raise "No valid points after filtering polygon points!"
    return polygon_points[valid_points]


# ----- Eye Orientation -----
def run_eye_orientation(eyeball_array, eccentricity_threshold: float = 0.1) -> float:
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
        orientation = 0.5 * np.arctan(
            2 * moments["mu11"] / (moments["mu20"] - moments["mu02"])
        )
        if (moments["mu20"] - moments["mu02"]) < 0:
            orientation += np.pi / 2
        orientation = np.mod(orientation + np.pi / 2, np.pi) - np.pi / 2

    return orientation


def eccentricity(moments: Dict[str, float]) -> float:
    if moments["mu20"] + moments["mu02"] == 0:
        return 1.0
    eccentricity = (
        (moments["mu20"] - moments["mu02"]) ** 2 + 4 * moments["mu11"] ** 2
    ) / (moments["mu20"] + moments["mu02"]) ** 2

    return eccentricity


# ----- Eye Center Estimation -----
def run_eye_center_estimation(
    pupil_array, iris_array
) -> Tuple[float, float, float, float]:
    pupil_diameter = float(
        np.linalg.norm(pupil_array[:, None, :] - pupil_array[None, :, :], axis=-1).max()
    )
    iris_diameter = float(
        np.linalg.norm(iris_array[:, None, :] - iris_array[None, :, :], axis=-1).max()
    )
    pupil_center_x, pupil_center_y = find_center_coords(pupil_array, pupil_diameter)
    iris_center_x, iris_center_y = find_center_coords(iris_array, iris_diameter)

    return pupil_center_x, pupil_center_y, iris_center_x, iris_center_y


def find_center_coords(
    polygon: np.ndarray,
    diameter: float,
    min_distance_between_sector_points: float = 0.75,
) -> Tuple[float, float]:
    min_distance_between_sector_points_in_px = (
        min_distance_between_sector_points * diameter
    )
    first_bisectors_point, second_bisectors_point = calculate_perpendicular_bisectors(
        polygon, min_distance_between_sector_points_in_px
    )
    return find_best_intersection(first_bisectors_point, second_bisectors_point)


def calculate_perpendicular_bisectors(
    polygon: np.ndarray,
    min_distance_between_sector_points_in_px: float,
    num_bisectors: int = 100,
    max_iterations: int = 50,
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
        bisectors_first_points = np.vstack(
            [bisectors_first_points, first_drawn_points[mask]]
        )
        bisectors_second_points = np.vstack(
            [bisectors_second_points, second_drawn_points[mask]]
        )
        if len(bisectors_first_points) >= num_bisectors:
            break
    else:
        "Not able to find enough random pairs of points on the arc with a large enough distance!"

    bisectors_first_points = bisectors_first_points[:num_bisectors]
    bisectors_second_points = bisectors_second_points[:num_bisectors]
    bisectors_center = (bisectors_first_points + bisectors_second_points) / 2
    inv_bisectors_center_slope = np.fliplr(
        bisectors_second_points - bisectors_first_points
    )
    inv_bisectors_center_slope[:, 1] = -inv_bisectors_center_slope[:, 1]
    norm = np.linalg.norm(inv_bisectors_center_slope, axis=1)
    inv_bisectors_center_slope[:, 0] /= norm
    inv_bisectors_center_slope[:, 1] /= norm
    first_bisectors_point = bisectors_center - inv_bisectors_center_slope
    second_bisectors_point = bisectors_center + inv_bisectors_center_slope

    return first_bisectors_point, second_bisectors_point


def find_best_intersection(
    fst_points: np.ndarray, sec_points: np.ndarray
) -> Tuple[float, float]:
    norm_bisectors = (sec_points - fst_points) / np.linalg.norm(
        sec_points - fst_points, axis=1
    )[:, np.newaxis]
    projections = (
        np.eye(norm_bisectors.shape[1])
        - norm_bisectors[:, :, np.newaxis] * norm_bisectors[:, np.newaxis]
    )
    R = projections.sum(axis=0)
    q = (projections @ fst_points[:, :, np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    intersection_x, intersection_y = p
    return intersection_x.item(), intersection_y.item()


def cartesian2polar(
    xs: np.ndarray, ys: np.ndarray, center_x: float, center_y: float
) -> Tuple[np.ndarray, np.ndarray]:
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


# ----- Geometry Estimation -----
def run_geometry_estimation(
    pupil_array,
    iris_array,
    pupil_x,
    pupil_y,
    iris_x,
    iris_y,
    algorithm_switch_std_threshold: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = iris_array[:, 0], iris_array[:, 1]
    rhos, _ = cartesian2polar(xs, ys, iris_x, iris_y)
    estimated_pupil, estimated_iris = circle_extrapolation(
        pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y
    )
    radius_std = rhos.std()
    if radius_std > algorithm_switch_std_threshold:
        _, estimated_iris = ellipse_fit(pupil_array, iris_array)

    return estimated_pupil, estimated_iris


# Circle extrapolation
def circle_extrapolation(
    pupil_array, iris_array, pupil_x, pupil_y, iris_x, iris_y
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimated_pupil = estimate(pupil_array, (pupil_x, pupil_y))
    estimated_iris = estimate(iris_array, (iris_x, iris_y))
    return estimated_pupil, estimated_iris


def estimate(
    vertices: np.ndarray, center_xy: Tuple[float, float], dphi: float = 0.9
) -> np.ndarray:
    rhos, phis = cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)
    padded_rhos = np.concatenate([rhos, rhos, rhos])
    padded_phis = np.concatenate([phis - 2 * np.pi, phis, phis + 2 * np.pi])
    interpolated_phis = np.arange(
        padded_phis.min(), padded_phis.max(), np.radians(dphi)
    )
    interpolated_rhos = np.interp(
        interpolated_phis, xp=padded_phis, fp=padded_rhos, period=2 * np.pi
    )
    mask = (interpolated_phis >= 0) & (interpolated_phis < 2 * np.pi)
    interpolated_phis, interpolated_rhos = (
        interpolated_phis[mask],
        interpolated_rhos[mask],
    )
    xs, ys = polar2cartesian(interpolated_rhos, interpolated_phis, *center_xy)
    estimated_vertices = np.column_stack([xs, ys])
    return estimated_vertices


# Ellipse fit
def ellipse_fit(pupil_array, iris_array) -> Tuple[np.ndarray, np.ndarray]:
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
    extrapolated_polygon = np.flip(
        np.roll(extrapolated_polygon, roll_amount, axis=0), axis=0
    )
    return extrapolated_polygon


def find_correspondence(src_point: np.ndarray, dst_points: np.ndarray) -> int:
    src_x, src_y = src_point
    distance = (dst_points[:, 1] - src_y) ** 2 + (dst_points[:, 0] - src_x) ** 2
    idx = np.where(distance == distance.min())[0]
    return idx


def parametric_ellipsis(
    a: float, b: float, x0: float, y0: float, theta: float, nb_step: int = 100
) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, nb_step)
    x_coords = x0 + b * np.cos(t) * np.sin(-theta) + a * np.sin(t) * np.cos(-theta)
    y_coords = y0 + b * np.cos(t) * np.cos(-theta) - a * np.sin(t) * np.sin(-theta)
    return np.array([x_coords, y_coords]).T


# ----- Linear Normalization -----
def run_linear_normalization(
    image,
    noise_mask,
    pupil_array,
    iris_array,
    eyeball_array,
    angle,
    oversat_threshold: PositiveInt = 254,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pupil_array) != len(iris_array):
        raise "The number of extrapolated iris and pupil points must be the same."
    pupil_points, iris_points = correct_orientation(
        pupil_array,
        iris_array,
        angle,
    )
    iris_mask = generate_iris_mask(iris_array, eyeball_array, noise_mask)

    iris_mask[image >= oversat_threshold] = False
    src_points = generate_correspondences(pupil_points, iris_points)
    normalized_image, normalized_mask = normalize_all(image, iris_mask, src_points)

    return to_uint8(normalized_image), normalized_mask


def generate_correspondences(
    pupil_points: np.ndarray, iris_points: np.ndarray, res_in_r: PositiveInt = 128
) -> np.ndarray:
    src_points = np.array(
        [
            pupil_points + x * (iris_points - pupil_points)
            for x in np.linspace(0.0, 1.0, res_in_r)
        ]
    )
    return np.round(src_points).astype(int)


def to_uint8(image: np.ndarray) -> np.ndarray:
    out_image = np.round(image * 255)
    out_image = out_image.astype(np.uint8)

    return out_image


def correct_orientation(
    pupil_points: np.ndarray, iris_points: np.ndarray, eye_orientation: float
) -> Tuple[np.ndarray, np.ndarray]:
    orientation_angle = np.degrees(eye_orientation)
    num_rotations = -round(orientation_angle * len(pupil_points) / 360.0)

    pupil_points = np.roll(pupil_points, num_rotations, axis=0)
    iris_points = np.roll(iris_points, num_rotations, axis=0)

    return pupil_points, iris_points


def generate_iris_mask(iris_array, eyeball_array, noise_mask: np.ndarray) -> np.ndarray:
    img_h, img_w = noise_mask.shape[:2]

    iris_mask = contour_to_mask(iris_array, (img_w, img_h))
    eyeball_mask = contour_to_mask(eyeball_array, (img_w, img_h))

    iris_mask = iris_mask & eyeball_mask
    iris_mask = ~(iris_mask & noise_mask) & iris_mask

    return iris_mask


def contour_to_mask(vertices: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
    width, height = mask_shape
    mask = np.zeros(shape=(height, width, 3))
    vertices = np.round(vertices).astype(np.int32)
    cv2.fillPoly(mask, pts=[vertices], color=(255, 0, 0))
    mask = mask[..., 0]
    mask = mask.astype(bool)

    return mask


def normalize_all(
    image: np.ndarray,
    iris_mask: np.ndarray,
    src_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src_shape = src_points.shape[0:2]
    src_points = np.vstack(src_points)
    image_size = image.shape
    src_points[src_points[:, 0] >= image_size[1], 0] = -1
    src_points[src_points[:, 1] >= image_size[0], 1] = -1
    normalized_image = np.array(
        [
            image[image_xy[1], image_xy[0]] if min(image_xy) >= 0 else 0
            for image_xy in src_points
        ]
    )
    normalized_image = np.reshape(normalized_image, src_shape)
    normalized_mask = np.array(
        [
            iris_mask[image_xy[1], image_xy[0]] if min(image_xy) >= 0 else False
            for image_xy in src_points
        ]
    )
    normalized_mask = np.reshape(normalized_mask, src_shape)

    return normalized_image / 255.0, normalized_mask


# ----- Filter Bank -----
def run_filter_blank(
    normalized_image, normalized_mask
) -> Tuple[np.ndarray, np.ndarray]:
    iris_responses: List[np.ndarray] = []
    mask_responses: List[np.ndarray] = []

    # First Gabor Filter
    kernel_size = (41, 21)
    sigma_phi = 7
    sigma_rho = 6.13
    theta_degrees = 90.0
    lambda_phi = 28
    iris_response, mask_response = convolve(
        kernel_size,
        sigma_phi,
        sigma_rho,
        theta_degrees,
        lambda_phi,
        normalized_image,
        normalized_mask,
    )
    iris_responses.append(iris_response)
    mask_responses.append(mask_response)

    # Second Gabor Filter
    kernel_size = (17, 21)
    sigma_phi = 2
    sigma_rho = 5.86
    theta_degrees = 90.0
    lambda_phi = 8
    iris_response, mask_response = convolve(
        kernel_size,
        sigma_phi,
        sigma_rho,
        theta_degrees,
        lambda_phi,
        normalized_image,
        normalized_mask,
    )
    iris_responses.append(iris_response)
    mask_responses.append(mask_response)

    return iris_responses, mask_responses


def convolve(
    kernel_size,
    sigma_phi,
    sigma_rho,
    theta_degrees,
    lambda_phi,
    normalized_image,
    normalized_mask,
) -> Tuple[np.ndarray, np.ndarray]:
    kernel_values = compute_kernel_values(
        kernel_size, sigma_phi, sigma_rho, theta_degrees, lambda_phi
    )
    kernel_norm = (
        np.linalg.norm(kernel_values.real, ord="fro")
        + np.linalg.norm(kernel_values.imag, ord="fro") * 1j
    )
    n_rows = 16
    n_cols = 256
    rhos, phis = generate_schema(n_rows, n_cols)
    i_rows, i_cols = normalized_image.shape
    k_rows, k_cols = kernel_values.shape
    p_rows = k_rows // 2
    p_cols = k_cols // 2
    iris_response = np.zeros((n_rows, n_cols), dtype=np.complex64)
    mask_response = np.zeros((n_rows, n_cols), dtype=np.complex64)
    padded_iris = polar_img_padding(normalized_image, p_rows, p_cols)
    padded_mask = polar_img_padding(normalized_mask, p_rows, p_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            pos = i * n_cols + j
            r_probe = min(round(rhos[pos] * i_rows), i_rows - 1)
            c_probe = min(round(phis[pos] * i_cols), i_cols - 1)
            iris_patch = padded_iris[
                r_probe : r_probe + k_rows, c_probe : c_probe + k_cols
            ]
            mask_patch = padded_mask[
                r_probe : r_probe + k_rows, c_probe : c_probe + k_cols
            ]
            non_padded_k_rows = (
                k_rows
                if np.logical_and(r_probe > p_rows, r_probe <= i_rows - p_rows)
                else (k_rows - max(p_rows - r_probe, r_probe + p_rows - i_rows))
            )
            iris_response[i][j] = (
                (iris_patch * kernel_values).sum() / non_padded_k_rows / k_cols
            )
            mask_response[i][j] = (
                0
                if iris_response[i][j] == 0
                else (mask_patch.sum() / non_padded_k_rows / k_cols)
            )
    iris_response.real = iris_response.real / kernel_norm.real
    iris_response.imag = iris_response.imag / kernel_norm.imag
    mask_response.imag = mask_response.real
    return iris_response, mask_response


def compute_kernel_values(
    kernel_size,
    sigma_phi,
    sigma_rho,
    theta_degrees,
    lambda_phi,
    dc_correction: bool = True,
    to_fixpoints: bool = True,
) -> np.ndarray:

    x, y = get_xy_mesh(kernel_size)
    rotx, roty = rotate(x, y, theta_degrees)
    carrier = 1j * 2 * np.pi / lambda_phi * rotx
    envelope = -(rotx**2 / sigma_phi**2 + roty**2 / sigma_rho**2) / 2
    kernel_values = np.exp(envelope + carrier)
    kernel_values /= 2 * np.pi * sigma_phi * sigma_rho

    if dc_correction:
        g_mean = np.mean(np.real(kernel_values), axis=-1)
        correction_term_mean = np.mean(envelope, axis=-1)
        kernel_values = (
            kernel_values - (g_mean / correction_term_mean)[:, np.newaxis] * envelope
        )

    kernel_values = normalize_kernel_values(kernel_values)
    if to_fixpoints:
        kernel_values = convert_to_fixpoint_kernelvalues(kernel_values)

    return kernel_values


def get_xy_mesh(kernel_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    ksize_phi_half = kernel_size[0] // 2
    ksize_rho_half = kernel_size[1] // 2
    y, x = np.meshgrid(
        np.arange(-ksize_phi_half, ksize_phi_half + 1),
        np.arange(-ksize_rho_half, ksize_rho_half + 1),
        indexing="xy",
        sparse=True,
    )

    return x, y


def rotate(x: np.ndarray, y: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    cos_theta = np.cos(angle * np.pi / 180)
    sin_theta = np.sin(angle * np.pi / 180)
    rotx = x * cos_theta + y * sin_theta
    roty = -x * sin_theta + y * cos_theta

    return rotx, roty


def normalize_kernel_values(kernel_values: np.ndarray) -> np.ndarray:
    norm_real = np.linalg.norm(kernel_values.real, ord="fro")
    if norm_real > 0:
        kernel_values.real /= norm_real
    norm_imag = np.linalg.norm(kernel_values.imag, ord="fro")
    if norm_imag > 0:
        kernel_values.imag /= norm_imag

    return kernel_values


def convert_to_fixpoint_kernelvalues(kernel_values: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(kernel_values):
        kernel_values.real = np.round(kernel_values.real * 2**15)
        kernel_values.imag = np.round(kernel_values.imag * 2**15)
    else:
        kernel_values = np.round(kernel_values * 2**15)

    return kernel_values


def polar_img_padding(img: np.ndarray, p_rows: int, p_cols: int) -> np.ndarray:
    i_rows, i_cols = img.shape
    padded_image = np.zeros((i_rows + 2 * p_rows, i_cols + 2 * p_cols))

    padded_image[p_rows : i_rows + p_rows, p_cols : i_cols + p_cols] = img
    padded_image[p_rows : i_rows + p_rows, 0:p_cols] = img[:, -p_cols:]
    padded_image[p_rows : i_rows + p_rows, -p_cols:] = img[:, 0:p_cols]

    return padded_image


def generate_schema(
    n_rows,
    n_cols,
    boundary_rho: List[float] = [0, 0.0625],
    # boundary_phi: "periodic-left",
    image_shape: Optional[List[PositiveInt]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.linspace(0 + boundary_rho[0], 1 - boundary_rho[1], n_rows, endpoint=True)
    phi = np.linspace(0, 1, n_cols, endpoint=False)
    phis, rhos = np.meshgrid(phi, rho)
    rhos = rhos.flatten()
    phis = phis.flatten()

    if image_shape is not None:
        rhos_pixel_values = rhos * image_shape[0]
        phis_pixel_values = phis * image_shape[1]
        rho_pixel_values = np.logical_or(
            np.less_equal(rhos_pixel_values % 1, 10 ** (-10)),
            np.less_equal(1 - 10 ** (-10), rhos_pixel_values % 1),
        ).all()
        phi_pixel_values = np.logical_or(
            np.less_equal(phis_pixel_values % 1, 10 ** (-10)),
            np.less_equal(1 - 10 ** (-10), phis_pixel_values % 1),
        ).all()
        if not rho_pixel_values:
            raise f"Choice for n_rows {n_rows} leads to interpolation errors, please change input variables"

        if not phi_pixel_values:
            raise f"Choice for n_cols {n_cols} leads to interpolation errors"
    return rhos, phis


# ----- Iris Encoder -----
def run_iris_encoder(
    iris_responses, mask_responses, mask_threshold: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iris_codes: List[np.ndarray] = []
    mask_codes: List[np.ndarray] = []
    for iris_response, mask_response in zip(iris_responses, mask_responses):
        iris_code = np.stack([iris_response.real > 0, iris_response.imag > 0], axis=-1)
        mask_code = np.stack(
            [
                mask_response.real >= mask_threshold,
                mask_response.imag >= mask_threshold,
            ],
            axis=-1,
        )
        iris_codes.append(iris_code)
        mask_codes.append(mask_code)

    return iris_codes, mask_codes


def pipeline(name) -> Tuple[np.ndarray, np.ndarray]:
    ir_image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    # Segmentation
    nn_input = preprocess_segmap(ir_image, INPUT_RESOLUTION, INPUT_CHANNELS).astype(
        np.float32
    )
    segmap = run_segmentation(nn_input)
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

    # Filter Bank
    iris_responses, mask_responses = run_filter_blank(normalized_image, normalized_mask)

    # Iris Encoder
    iris_codes, mask_codes = run_iris_encoder(iris_responses, mask_responses)

    return iris_codes, mask_codes


if __name__ == "__main__":
    iris_codes, mask_codes = pipeline(INPUT_IMAGE)
    iris_codes2, mask_codes2 = pipeline(INPUT_IMAGE2)
    iris_codes_other, mask_codes_other = pipeline(INPUT_IMAGE_OTHER)

    match_dist, _ = matching.hamming_distance(
        iris_codes, mask_codes, iris_codes2, mask_codes2
    )
    match_dist_other, _ = matching.hamming_distance(
        iris_codes, mask_codes, iris_codes_other, mask_codes_other
    )

    print(match_dist)
    print(match_dist_other)
