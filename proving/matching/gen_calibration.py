import os
import json
import torch

model_path = os.path.join('network.onnx')
settings_path = os.path.join('settings.json')
cal_path = os.path.join("calibration.json")

shape = [2, 16, 256, 2]
data_array = (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data = [data_array])

json.dump(data, open(cal_path, 'w'))
