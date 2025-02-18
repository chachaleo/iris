
from typing import List, Optional, Tuple, Union

import numpy as np


def hamming_distance(
    iris_codes_a,
    mask_codes_a,
    iris_codes_b,
    mask_codes_b,

    rotation_shift = 15,
    normalise: bool = False,
    norm_mean: float = 0.45,
    norm_gradient: float = 0.00005,
    separate_half_matching: bool = False,
    weights: Optional[List[np.ndarray]] = None,
) -> Tuple[float, int]:
    half_codewidth = []

    for probe_code, gallery_code in zip(iris_codes_a, iris_codes_b):
        if probe_code.shape != gallery_code.shape:
            raise "probe and gallery iris codes are of different sizes"
        if (probe_code.shape[1] % 2) != 0:
            raise "number of columns of iris codes need to be even"
        if separate_half_matching:
            half_codewidth.append(int(probe_code.shape[1] / 2))

    if weights:
        for probe_code, w in zip(iris_codes_a, weights):
            if probe_code.shape != w.shape:
                raise "weights table and iris codes are of different sizes"

    # Calculate the Hamming distance between probe and gallery template.
    match_dist = 1
    match_rot = 0
    for current_shift in [0] + [y for x in range(1, rotation_shift + 1) for y in (-x, x)]:
        irisbits, maskbits = get_bitcounts(iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b, current_shift)
        totalirisbitcount, totalmaskbitcount = count_nonmatchbits(irisbits, maskbits, half_codewidth, weights)
        totalmaskbitcountsum = totalmaskbitcount.sum()
        if totalmaskbitcountsum == 0:
            continue

        if normalise:
            normdist = normalized_HD(totalirisbitcount.sum(), totalmaskbitcountsum, norm_mean, norm_gradient)
            if separate_half_matching:
                normdist0 = (
                    normalized_HD(totalirisbitcount[0], totalmaskbitcount[0], norm_mean, norm_gradient)
                    if totalmaskbitcount[0] > 0
                    else norm_mean
                )
                normdist1 = (
                    normalized_HD(totalirisbitcount[1], totalmaskbitcount[1], norm_mean, norm_gradient)
                    if totalmaskbitcount[0] > 0
                    else norm_mean
                )
                Hdist = np.mean(
                    [
                        normdist,
                        (normdist0 * totalmaskbitcount[0] + normdist1 * totalmaskbitcount[1]) / totalmaskbitcountsum,
                    ]
                )
            else:
                Hdist = normdist
        else:
            Hdist = totalirisbitcount.sum() / totalmaskbitcountsum

        if Hdist < match_dist:
            match_dist = Hdist
            match_rot = current_shift

    return match_dist, match_rot



def normalized_HD(irisbitcount: int, maskbitcount: int, norm_mean: float, norm_gradient: float) -> float:
    # Linear approximation to replace the previous sqrt-based normalization term.
    norm_HD = max(0, norm_mean - (norm_mean - irisbitcount / maskbitcount) * (norm_gradient * maskbitcount + 0.5))
    return norm_HD


def get_bitcounts(iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b, shift: int) -> np.ndarray:

    irisbits = [
        np.roll(probe_code, shift, axis=1) != gallery_code
        for probe_code, gallery_code in zip(iris_codes_a, iris_codes_b)
    ]
    maskbits = [
        np.roll(probe_code, shift, axis=1) & gallery_code
        for probe_code, gallery_code in zip(mask_codes_a, mask_codes_b)
    ]
    return irisbits, maskbits


def count_nonmatchbits(
    irisbits: np.ndarray,
    maskbits: np.ndarray,
    half_width: Optional[List[int]] = None,
    weights: Optional[List[np.ndarray]] = None,
) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
    if weights:
        irisbitcount = [np.sum((x & y) * z, axis=0) / z.sum() * z.size for x, y, z in zip(irisbits, maskbits, weights)]
        maskbitcount = [np.sum(y * z, axis=0) / z.sum() * z.size for y, z in zip(maskbits, weights)]
    else:
        irisbitcount = [np.sum(x & y, axis=0) for x, y in zip(irisbits, maskbits)]
        maskbitcount = [np.sum(y, axis=0) for y in maskbits]

    if half_width:
        totalirisbitcount = np.sum(
            [[np.sum(x[hw:, ...]), np.sum(x[:hw, ...])] for x, hw in zip(irisbitcount, half_width)], axis=0
        )
        totalmaskbitcount = np.sum(
            [[np.sum(y[hw:, ...]), np.sum(y[:hw, ...])] for y, hw in zip(maskbitcount, half_width)], axis=0
        )
    else:
        totalirisbitcount = np.sum(irisbitcount)
        totalmaskbitcount = np.sum(maskbitcount)

    return totalirisbitcount, totalmaskbitcount