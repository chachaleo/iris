
from typing import List, Tuple, Union

import numpy as np


def hamming_distance(
    iris_codes_a,
    mask_codes_a,
    iris_codes_b,
    mask_codes_b,

    rotation_shift = 15,
) -> Tuple[float, int]:
    for probe_code, gallery_code in zip(iris_codes_a, iris_codes_b):
        if probe_code.shape != gallery_code.shape:
            raise "probe and gallery iris codes are of different sizes"
        if (probe_code.shape[1] % 2) != 0:
            raise "number of columns of iris codes need to be even"

    # Calculate the Hamming distance between probe and gallery template.
    match_dist = 1
    match_rot = 0

    for current_shift in [0] + [y for x in range(1, rotation_shift + 1) for y in (-x, x)]:
        irisbits, maskbits = get_bitcounts(iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b, current_shift)

        totalirisbitcount, totalmaskbitcount = count_nonmatchbits(irisbits, maskbits)

        totalmaskbitcountsum = totalmaskbitcount.sum()
        if totalmaskbitcountsum == 0:
            continue

        Hdist = totalirisbitcount.sum() / totalmaskbitcountsum

        if Hdist < match_dist:
            match_dist = Hdist
            match_rot = current_shift

    return match_dist, match_rot


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
) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
    irisbitcount = [np.sum(x & y, axis=0) for x, y in zip(irisbits, maskbits)]
    maskbitcount = [np.sum(y, axis=0) for y in maskbits]

    totalirisbitcount = np.sum(irisbitcount)
    totalmaskbitcount = np.sum(maskbitcount)

    return totalirisbitcount, totalmaskbitcount