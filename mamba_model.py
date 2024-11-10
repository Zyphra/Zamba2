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
        num_mem_blocks =2
        mamba_ngroups = 1
        use_shared_attention_lora = False
        json_config = load_config_hf(model_name)
        state_dict = load_state_dict_hf(model_name)
        if "num_mem_blocks" in json_config.keys():
            num_mem_blocks = json_config["num_mem_blocks"]
        if "mamba_ngroups" in json_config.keys():
            mamba_ngroups = json_config["mamba_ngroups"]
        if "use_shared_attention_lora" in json_config.keys():
            use_shared_attention_lora = json_config["use_shared_attention_lora"]
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
        mamba_ngroups = mamba_ngroups,
        layer_mapping = json_config["layers_block_type"],
        add_bias_linear = json_config["add_bias_linear"],
        use_shared_block_lora = json_config["use_shared_block_lora"],
        lora_rank = json_config["lora_rank"],
        gated_linear_unit = json_config["gated_linear_unit"],
        kv_channels = json_config["kv_channels"],
        ffn_hidden_size = json_config["ffn_hidden_size"],
        vocab_size = json_config["vocab_size"],
        num_mem_blocks = num_mem_blocks,
        #num_key_value_heads = json_config["num_key_value_heads"],
        num_query_groups = json_config["num_query_groups"],
        use_shared_attention_lora = use_shared_attention_lora,
        use_mem_rope = json_config["use_mem_rope"],
        )
        if model_name != "Zyphra/Zamba2-2.7B":
            g_indices = []
            for i,el in enumerate(json_config["layers_block_type"]):
                if el == "g":
                    g_indices.append(i)
            i = 0
            #die
            for k in list(state_dict.keys()):
                new_k = k.replace("model","decoder").replace("mamba_layers","layers").replace("mamba","mixer").replace("input_layernorm","norm").replace("linear_layers","block_map").replace("feed_forward","mlp.mixer").replace("self_attn.o_proj","sa.mixer.linear_proj").replace("pre_ff_layernorm","mlp.norm").replace("in_proj","in_proj.0").replace("self_attn.linear_q","sa.mixer.linear_q").replace("self_attn.linear_k","sa.mixer.linear_k").replace("self_attn.linear_v","sa.mixer.linear_v")
                #print("NUM MEM BLOCKS: ", num_mem_blocks)
                for i in range(num_mem_blocks):
                    new_k = new_k.replace("decoder.blocks." + str(i) + ".norm.weight","decoder.blocks." + str(i) + ".sa.norm.weight")
                #new_k = new_k.replace("decoder.blocks.1.norm.weight","decoder.blocks.1.sa.norm.weight")
                if "block_map" in new_k:
                    block_idx = int(new_k.split("block_map.")[1].split(".")[0])
                    i +=1
                    new_idx = g_indices[block_idx]
                    new_k = new_k.replace(str(block_idx), str(new_idx))
                state_dict[new_k] = state_dict[k]
                del state_dict[k]
            state_dict["embedding.weight"] = state_dict["decoder.embed_tokens.weight"]
            del state_dict["decoder.embed_tokens.weight"]

            # merge QKV together manually
            for i in range(num_mem_blocks): 
                q = state_dict["decoder.blocks." + str(i) + ".self_attn.q_proj.weight"]
                k = state_dict["decoder.blocks." + str(i) + ".self_attn.k_proj.weight"]
                v = state_dict["decoder.blocks." + str(i) + ".self_attn.v_proj.weight"]
                qkv = torch.concat([q,k,v], dim=0)
                state_dict["decoder.blocks."+str(i)+".sa.mixer.linear_qkv.weight"] = qkv
                del state_dict["decoder.blocks." + str(i) + ".self_attn.q_proj.weight"]
                del state_dict["decoder.blocks." + str(i) + ".self_attn.k_proj.weight"]
                del state_dict["decoder.blocks." + str(i) + ".self_attn.v_proj.weight"]
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