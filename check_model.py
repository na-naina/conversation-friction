from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Model type: {type(model).__name__}")

# Try to find layers
m = model
for attr in ['model', 'language_model', 'text_model']:
    if hasattr(m, attr):
        sub = getattr(m, attr)
        print(f"\nmodel.{attr}: {type(sub).__name__}")
        if hasattr(sub, 'layers'):
            print(f"  -> .layers: {len(sub.layers)} layers")
        if hasattr(sub, 'model') and hasattr(sub.model, 'layers'):
            print(f"  -> .model.layers: {len(sub.model.layers)} layers")
