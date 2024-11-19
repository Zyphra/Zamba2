from mamba_model import MambaModel
from mamba_config import MambaConfig
import torch
from transformers import AutoTokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


#tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-2.7B")
tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B-Instruct")
#tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B")
input_text = 'A funny prompt would be '
input_text = '<im_start>user\nWhat is the meaning of life?<im_end>\n<im_start>assistant:\n'
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")["input_ids"].transpose(0,1)
#model = MambaModel.from_pretrained(model_name = "Zyphra/Zamba2-2.7B").cuda().half()
#model = MambaModel.from_pretrained(model_name = "Zyphra/Zamba2-1.2B").cuda().half()
model = MambaModel.from_pretrained(model_name = "Zyphra/Zamba2-7B-Instruct").cuda().half()
tokens_to_generate = 200

#model_hf = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-1.2B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16)

model.eval()
with torch.no_grad():
    for _ in range(tokens_to_generate):
        out = model(input_ids)
        out_last = out[:, -1]
        idx = torch.argmax(out_last)[None, None]
        input_ids = torch.cat((input_ids, idx), dim=0)
input_ids = input_ids.transpose(0, 1)[0]
print(repr(tokenizer.decode(input_ids.cpu().numpy().tolist())))
