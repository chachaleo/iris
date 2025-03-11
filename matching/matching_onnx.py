import torch
import torch.nn as nn
import torch.onnx


import torch
import torch.nn as nn

class HammingDistanceModel(nn.Module):
    def __init__(self, rotation_shift=15):
        super(HammingDistanceModel, self).__init__()
        self.rotation_shift = rotation_shift

    def forward(self, iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b):
        rotation_shift = self.rotation_shift
        init = True

        for current_shift in [0] + [y for x in range(1, rotation_shift + 1) for y in (-x, x)]:
            irisbits, maskbits = get_bitcounts(iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b, current_shift)
            totalirisbitcount, totalmaskbitcount = count_nonmatchbits(irisbits, maskbits)

            totalmaskbitcountsum = totalmaskbitcount.sum()
            if totalmaskbitcountsum == torch.zero_:
                continue

            Hdist = totalirisbitcount.sum() / totalmaskbitcountsum

            if init :
                match_dist = torch.minimum(torch.ones(1), Hdist)
                init = False
            else :
                match_dist = torch.minimum(match_dist, Hdist)

        return match_dist

def get_bitcounts(iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b, shift: int):
    irisbits = [
        torch.roll(torch.tensor(probe_code), shifts=shift, dims=1) != torch.tensor(gallery_code)
        for probe_code, gallery_code in zip(iris_codes_a, iris_codes_b)
    ]
    maskbits = [
        torch.roll(torch.tensor(probe_code), shifts=shift, dims=1) & torch.tensor(gallery_code)
        for probe_code, gallery_code in zip(mask_codes_a, mask_codes_b)
    ]
    return irisbits, maskbits

def count_nonmatchbits(irisbits, maskbits):

    irisbitcount = [torch.sum(x & y, axis=0) for x, y in zip(irisbits, maskbits)]
    maskbitcount = [torch.sum(y, axis=0) for y in maskbits]

    totalirisbitcount = torch.stack(irisbitcount).sum()
    totalmaskbitcount = torch.stack(maskbitcount).sum()

    return totalirisbitcount, totalmaskbitcount



batch_size = 16
code_length = 256
iris_codes_a = torch.randint(2, (2, batch_size, code_length, 2), dtype=torch.bool)
mask_codes_a = torch.randint(2, (2, batch_size, code_length, 2), dtype=torch.bool)
iris_codes_b = torch.randint(2, (2, batch_size, code_length, 2), dtype=torch.bool)
mask_codes_b = torch.randint(2, (2, batch_size, code_length, 2), dtype=torch.bool)

model = HammingDistanceModel(rotation_shift=15)
onnx_path = "../onnx/hamming_distance.onnx"

torch.onnx.export(
    model,
    (iris_codes_a, mask_codes_a, iris_codes_b, mask_codes_b),
    onnx_path,
    export_params=True,
    opset_version=14,
    input_names=["iris_codes_a", "mask_codes_a", "iris_codes_b", "mask_codes_b"],
    output_names=["match_dist"],
)

print(f"ONNX model saved to {onnx_path}")
