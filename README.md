# Zamba2-2.7B

Zamba2-2.7B is a hybrid model between state-space models and transformers. It broadly follows the [Zamba architecture](https://arxiv.org/abs/2405.16712) which consists of a Mamba backbone alternating with shared transformer blocks. Zamba-2-2.7B possesses three major improvements over Zamba1:

1.) Mamba1 blocks have been replaced with Mamba2 blocks.
2.) Instead of a single shared attention block, we utilize two shared attention blocks which are interleaved in an ABAB pattern through the network.
3.) We apply a LoRA projector to each shared MLP block allowing the network to specialize the MLPs at each shared layer with a minimal increase in total parameter count.

Zamba was trained using next-token prediction. It uses the Mistral v0.1 tokenizer. Zamba2-2.7B was pre-trained on 3T tokens of text and code data sourced from open web-datasets, including [Zyda](https://arxiv.org/abs/2406.01981). Subsequently, in a second phase, Zamba2-2.7B was annealed on a mixture of 100B high-quality tokens.

This is the standalone Pytorch implementation of Zamba2-2.7B. A Huggingface-compatible version may be found [here](https://huggingface.co/Zyphra/Zamba2-2.7B)

## Quick start

### Prerequisites

To begin, clone and install this repo:

1.) `git clone https://github.com/Zyphra/Zamba2.git`

2.) cd `Zamba2`

3.) Install the repository: `pip install -e .`

4.) Install core mamba dependencies `pip install -U mamba-ssm causal-conv1d`


You can run the model without using the optimized Mamba kernels, but it is **not** recommended as it will result in significantly higher latency and memory usage. 

### Inference

```python
from mamba_model import MambaModel
from mamba_config import MambaConfig
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-2.7B")
input_text = 'A funny prompt would be '
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")["input_ids"].transpose(0,1)
model = MambaModel.from_pretrained(model_name = "Zyphra/Zamba2-2.7B").cuda().half()
tokens_to_generate = 20
model.eval()
with torch.no_grad():
    for _ in range(tokens_to_generate):
        out = model(input_ids)
        out_last = out[:, -1]
        idx = torch.argmax(out_last)[None, None]
        input_ids = torch.cat((input_ids, idx), dim=0)
input_ids = input_ids.transpose(0, 1)[0]
print(repr(tokenizer.decode(input_ids.cpu().numpy().tolist())))
```

## Model Details

Zamba2-2.7B utilizes and extends our original Zamba hybrid SSM-attention architecture. The core Zamba architecture consists of a backbone of Mamba layers interleaved with one or more shared attention layers (one shared attention in Zamba1, two in Zamba2). This attention has shared weights to minimize the parameter cost of the model. We find that concatenating the original model embeddings to the input to this attention block improves performance, likely due to better maintenance of information across depth. The Zamba2 architecture also applies LoRA projection matrices to the shared MLP to gain some additional expressivity in each block and allow each shared block to specialize slightly to its own unique position while keeping the additional parameter overhead small. 

<center>
<img src="https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/XrEIEBxd0fqIgh3LyArAV.png" width="300" alt="Zamba architecture">
</center>


## Performance

Zamba2-2.7B achieves leading and state-of-the-art performance among models of <3B parameters and is competitive with some models of significantly greater size. Moreover, due to its unique hybrid SSM architecture, Zamba2-2.7B achieves extremely low inference latency and rapid generation with a significantly smaller memory footprint than comparable transformer based models. 

Zamba2-2.7B's high performance and small inference compute and memory footprint renders it an ideal generalist model for on-device applications.

<center>
<img src="https://cdn-uploads.huggingface.co/production/uploads/65c05e75c084467acab2f84a/U7VD9PYLj3XcEjgV08sP5.png" width="700" alt="Zamba performance">
</center>

<center>
<img src="https://cdn-uploads.huggingface.co/production/uploads/65bc13717c6ad1994b6619e9/3u8k7tcRi-oC_ltGhdHAk.png" width="800" alt="Zamba evals">
</center>

Time to First Token (TTFT)             |  Output Generation
:-------------------------:|:-------------------------:
![](https://cdn-uploads.huggingface.co/production/uploads/65bc13717c6ad1994b6619e9/BmE8X6tDNVw5OJcbZt8sZ.png)  |  ![](https://cdn-uploads.huggingface.co/production/uploads/65bc13717c6ad1994b6619e9/wECc9cItK1FW1MOMGSLrp.png)


<center>
<img src="https://cdn-uploads.huggingface.co/production/uploads/65bc13717c6ad1994b6619e9/nhoss41xlzfEBZzcQXI6z.png" width="700" alt="Zamba inference and memory cost">
</center>

## Notice

Zamba2-2.7B is a pretrained base model and therefore does not have any moderation mechanism and may output toxic or otherwise harmful language. In addition, one should not expect good instruct or chat performance, as this model was not fine-tuned for instruction following or chat.
