import json
import os

import torch

def load_config_hf(model_name):
    base_path = "/workspace/Zamba2-1.2B"
    config_path = os.path.join(base_path, "config.json")
    return json.load(open(config_path))

def load_state_dict_hf(model_name, device=None, dtype=None):
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    base_path = "/workspace/Zamba2-1.2B"
    weights_path = os.path.join(base_path, "pytorch_model.bin")
    return torch.load(weights_path, map_location=mapped_device)
