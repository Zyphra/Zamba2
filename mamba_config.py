from dataclasses import dataclass, field
from typing import Callable
from typing import List
import torch
import torch.nn.functional as F
from utils import init_method_normal, scaled_init_method_normal

@dataclass
class MambaConfig():

    # model architecture
    base_model_type: str = "mamba"
    num_layers: int = 0
    hidden_size: int = 0
    state_size: int = 0
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    expansion_factor: int = 2
    conv_dimension: int = 0
    conv_bias: bool = True
    bias: bool = True
    use_fast_path: bool = True
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    use_module_layernorm: bool = True 

    rms_norm: bool = False 
    fused_add_norm: bool = False  
    residual_in_fp32: bool = False
    hidden_dropout: float = 0.0
    ffn_hidden_size: int = None
    gated_linear_unit: bool = False
    kv_channels: int = None
    kv_mem_channels: int = None
    num_attention_heads: int = 0
    num_query_groups: int = None
    num_mem_query_groups: int = None
    attention_dropout: float = 0.1
    num_mem_heads: int = 0 
    use_mem_mlp: bool = False
    window_size: int = None
    gateconv_expansion_factor: int = 2
    layer_mapping: List[str] = field(default_factory=lambda: [""])
    vocab_size: int = 0
    device: str = "cuda"
    use_mem_rope: bool = False
    use_shared_block_lora: bool = False
    lora_rank: int = 16
    num_mem_blocks: int = 1
    use_low_rank_mamba_proj: bool = False
    num_shared_mamba_proj: int = 1
    mamba_lora_rank: int = 1
    use_shared_attention_lora: bool = False
    rope_theta: int = 10000
    
    

    fp32_residual_connection: bool = False
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    add_bias_linear: bool = False
    activation_func: Callable = F.gelu
    num_moe_experts: int = None

    # initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02

    # mixed-precision
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    gated_linear_unit: bool = False
    bias_gelu_fusion: bool = False 
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = False 

    # activation recomputation
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None


    def __post_init__(self):
        # Print initial configuration
        print("\n=== Initializing MambaConfig ===")
        print(f"Model type: {self.base_model_type}")
        print(f"Num layers: {self.num_layers}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"State size: {self.state_size}")
        print(f"Vocab size: {self.vocab_size}")

        # Force attention softmax to be in fp32 if query/key layer scaling is enabled
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        # Set default FFN hidden size if not provided
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size
            
        # Set default KV channels based on attention heads
        if self.kv_channels is None and self.num_attention_heads is not None:
            self.kv_channels = self.hidden_size // self.num_attention_heads
        
        # Set default memory KV channels based on memory heads
        if self.kv_mem_channels is None and self.num_mem_heads > 0:
            self.kv_mem_channels = self.hidden_size // self.num_mem_heads

        # Set default query groups if not specified
        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        # Set default memory query groups if not specified
        if self.num_mem_query_groups is None:
            self.num_mem_query_groups = self.num_mem_heads

        # Force attention softmax to be in fp32 if query/key layer scaling is enabled
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        # Validate bias gelu fusion settings
        if self.bias_gelu_fusion:
            if not self.add_bias_linear:
                raise ValueError(
                    "When bias_gelu_fusion is True, add_bias_linear must also be True."
                )

            if self.activation_func != F.gelu:
                raise ValueError(f'When bias_gelu_fusion is True, activation_func must be F.gelu.')

        # Set default initialization methods
        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )

        # Print derived configuration
        print(f"\nDerived configuration:")
        print(f"FFN hidden size: {self.ffn_hidden_size}")
        print(f"KV channels: {self.kv_channels}")
        print(f"Num query groups: {self.num_query_groups}")
        print("=== MambaConfig Initialization Complete ===\n")
