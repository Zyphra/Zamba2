import logging
from typing import Literal, Optional, Union
import functools
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import os
from mamba_block import MambaBlock, MambaDecoder
from mamba_config import MambaConfig
from hf_utils import *
import os, json
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1, 
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        initializer_cfg = None,
    ) -> None:
        super().__init__()

        self.config: MambaConfig = config
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        
        if self.pre_process:
            self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        self.decoder = MambaDecoder(
            config = self.config,
            pre_process = self.pre_process,
            post_process = self.post_process,
        )
        #if post_process:
            # self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = self.config.add_bias_linear)
            # if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
            #     self.initialize_last_stage_with_word_embeddings()
            
        self.apply(
            partial(
                _init_weights,
                n_layer=self.config.num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
    def initialize_last_stage_with_word_embeddings(self):
        with torch.no_grad():
            self.output_layer.weight = self.embedding.weight

    def forward(
        self,
        input_ids,
        position_ids = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
    ) -> Tensor:

        if decoder_input is not None:
            pass
        elif self.pre_process:
            
            decoder_input = self.embedding(input_ids)
            
            decoder_input = decoder_input.permute(1,0,2)
        else:
            decoder_input = None
            

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            residual=None,
            inference_params=inference_params,
        )
        
        
        if not self.post_process:
            return hidden_states
        
        logits = hidden_states @ self.embedding.weight.T
        return logits.contiguous()

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        NUM_MEM_BLOCKS =2
        json_config = load_config_hf(model_name)
        state_dict = load_state_dict_hf(model_name)
        config = MambaConfig(
        num_layers = json_config["num_hidden_layers"],
        hidden_size = json_config["hidden_size"],
        state_size = json_config["state_size"],
        conv_dimension = json_config["conv_dimension"],
        expansion_factor = json_config["expansion_factor"],
        rms_norm = True,
        use_mem_mlp = True,
        num_attention_heads = json_config["num_attention_heads"],
        num_mem_heads = json_config["num_attention_heads"],
        mamba_headdim = json_config["mamba_headdim"],
        layer_mapping = json_config["layers_block_type"],
        add_bias_linear = json_config["add_bias_linear"],
        use_shared_block_lora = json_config["use_shared_block_lora"],
        lora_rank = json_config["lora_rank"],
        gated_linear_unit = json_config["gated_linear_unit"],
        kv_channels = json_config["kv_channels"],
        ffn_hidden_size = json_config["ffn_hidden_size"],
        vocab_size = json_config["vocab_size"],
        num_mem_blocks = NUM_MEM_BLOCKS,
        )
        model = MambaModel(config = config, max_sequence_length = 4096)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)