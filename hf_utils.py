import json
import os
import torch

from transformers.utils import CONFIG_NAME
from transformers.utils.hub import cached_file

HF_TOKEN = os.getenv("HF_TOKEN")

def load_config_hf(model_name):
    resolved_archive_file = cached_file(
        model_name, 
        CONFIG_NAME, 
        token=HF_TOKEN,
        _raise_exceptions_for_missing_entries=True, 
        force_download=False
    )
    return json.load(open(resolved_archive_file))

def load_state_dict_hf(model_name, device=None, dtype=None):
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    WEIGHTS_NAME = "pytorch_model.bin"
    resolved_archive_file = cached_file(
        model_name, 
        WEIGHTS_NAME, 
        token=HF_TOKEN,
        _raise_exceptions_for_missing_entries=True, 
        force_download=False
    )
    return torch.load(resolved_archive_file, map_location=mapped_device)
