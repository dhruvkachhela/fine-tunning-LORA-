import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name for fine-tuning
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with float16 precision and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
