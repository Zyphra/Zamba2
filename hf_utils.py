import json

import torch

from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file
from safetensors.torch import load_file
import os


def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=True, force_download=False)
    return json.load(open(resolved_archive_file))

def load_state_dict_hf(model_name, device=None, dtype=None):
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    if model_name == "Zyphra/Zamba2-2.7B":
       weights_name = "Zamba2_2p7b_direct_from_pytorch.pt"
       resolved_archive_file = cached_file(model_name,weights_name, _raise_exceptions_for_missing_entries=True, force_download=False)
       return torch.load(resolved_archive_file, map_location=mapped_device)
    else:
        state_dict = {}
        archive_files = []
        if model_name == "Zyphra/Zamba2-7B":
            weight_names = ["model-00001-of-00003.safetensors","model-00002-of-00003.safetensors","model-00003-of-00003.safetensors"]
        elif model_name == "Zyphra/Zamba2-1.2B" or model_name == "Zyphra/Zamba2-1.2B-Instruct":
            weight_names = ["model.safetensors"]
        elif model_name == "Zyphra/Zamba2-7B-Instruct":
            weight_names = ["model-00001-of-00004.safetensors","model-00002-of-00004.safetensors","model-00003-of-00004.safetensors","model-00004-of-00004.safetensors"]
        elif model_name == "Zyphra/Zamba2-2.7B-Instruct":
            weight_names = ["model-00001-of-00002.safetensors","model-00002-of-00002.safetensors"]
        else:
            raise ValueError("Model name not recognized")

        for weight_name in weight_names:
            resolved_archive_file = cached_file(model_name,weight_name, _raise_exceptions_for_missing_entries=True, force_download=False)
            shard_state_dict = load_file(resolved_archive_file)
            state_dict.update(shard_state_dict)
            archive_files.append(resolved_archive_file)
        return state_dict
    